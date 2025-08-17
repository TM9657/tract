use tract_core::internal::TractErrorContext;
use tract_core::internal::{format_err, tract_ndarray};
use tract_tflite::tflite;
use tract_tflite::Tflite;
use tract_tflite::internal::TfliteProtoModel;
use tract_core::prelude::*;

fn main() -> TractResult<()> {
    let mut args = std::env::args();
    let _ = args.next(); // exe
    let path = args.next().expect("pass path to model.tflite");
    let image_path = args.next(); // optional

    println!("Loading model from: {}", path);
    let mut f = std::fs::File::open(&path)?;
    let t = Tflite::default();
    let proto = t.proto_model_for_read(&mut f).context("Reading proto model")?;

    // Diagnostic: dump tensors and operators from the proto to help debug translation issues.
    {
        let root = proto.root();
        let subgraphs = root.subgraphs().context("No subgraphs in Tflite model")?;
        let main = subgraphs.get(0);
        eprintln!("Proto diagnostic: tensors: {}", main.tensors().map(|v| v.len()).unwrap_or(0));
        if let Some(tensors) = main.tensors() {
            for (i, t) in tensors.iter().enumerate() {
                let name = t.name().unwrap_or("");
                let shape = t
                    .shape()
                    .map(|s| s.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","))
                    .unwrap_or_else(|| "<no-shape>".to_string());
                let dtype = format!("{:?}", t.type_());
                let buffer = t.buffer();
                eprintln!("tensor {}: '{}' type={} buffer={} shape=[{}]", i, name, dtype, buffer, shape);
                if buffer != 0 {
                    if let Some(buffers) = root.buffers() {
                        let b = buffers.get(buffer as usize);
                        if let Some(data) = b.data() {
                            let bytes = data.bytes();
                            let hexdump: Vec<String> = bytes.iter().take(16).map(|b| format!("{:02x}", b)).collect();
                            eprintln!(" buffer bytes (first up to 16): {}", hexdump.join(" "));
                            // try to interpret first 4/8/12/16 bytes as i32/f32 little endian
                            if bytes.len() >= 4 {
                                use std::convert::TryInto;
                                let v4: [u8;4] = bytes[0..4].try_into().unwrap();
                                let as_i32 = i32::from_le_bytes(v4);
                                let as_f32 = f32::from_le_bytes(v4);
                                eprintln!("  first32 i32={} f32={}", as_i32, as_f32);
                            }
                        }
                    }
                }
            }
        }
        eprintln!("Proto diagnostic: operators: {}", main.operators().map(|v| v.len()).unwrap_or(0));
        if let Some(ops) = main.operators() {
            for (i, op) in ops.iter().enumerate() {
                let inputs = op.inputs().map(|v| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")).unwrap_or_default();
                let outputs = op.outputs().map(|v| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")).unwrap_or_default();
                let builtin = format!("{:?}", op.builtin_options_type());
                eprintln!("op {}: opcode_index={} builtin_options_type={} inputs=[{}] outputs=[{}]", i, op.opcode_index(), builtin, inputs, outputs);
            }
        }
    }

    // Attempt full translation, print pretty error on failure
    let model = match t.model_for_proto_model(&proto) {
        Ok(m) => {
            println!("Model translated OK: {} nodes", m.nodes().len());
            m
        }
        Err(e) => {
            eprintln!("{:#}", e);
            std::process::exit(1);
        }
    };

    // If no image provided, we're done.
    let image_path = match image_path {
        Some(p) => p,
        None => return Ok(()),
    };

    // Load image
    let img = image::open(&image_path).map_err(|e| format_err!("Failed to open image: {}", e))?.to_rgb8();

    // Try to infer input shape and dtype from the translated model.
    // Default to NHWC 1x224x224x3 and f32 scaling to [0,1].
    let mut target_h = 224usize;
    let mut target_w = 224usize;
    let mut as_f32 = true;
    let mut input_dt: Option<DatumType> = None;
    if let Ok(fact) = model.input_fact(0) {
        if let Some(shape) = fact.shape.as_concrete() {
            let dims: TVec<usize> = shape.into();
            if dims.len() == 4 {
                // try NHWC (1,h,w,3)
                if dims[3] == 3 {
                    target_h = dims[1];
                    target_w = dims[2];
                } else if dims[1] == 3 {
                    // NCHW (1,3,h,w)
                    target_h = dims[2];
                    target_w = dims[3];
                }
            }
    }
        // Diagnostic: print input datum type and quantization params if present
        eprintln!("Input fact datatype: {:?}, is_quantized: {}", fact.datum_type, fact.datum_type.is_quantized());
        if fact.datum_type.is_quantized() {
            let (zp, scale) = fact.datum_type.zp_scale();
            eprintln!("Input qparams: zero_point={}, scale={}", zp, scale);
        }
    // decide whether to normalize to 0..1 depending on datum type
    // quantized / integer inputs should not be normalized to float in [0,1]
    as_f32 = fact.datum_type.is_float();
    input_dt = Some(fact.datum_type.clone());
    }

    let resized = image::imageops::resize(&img, target_w as u32, target_h as u32, ::image::imageops::FilterType::Triangle);
    let image_tensor: Tensor = match input_dt {
        Some(dt) if dt.is_quantized() => {
            let (zp, scale) = dt.zp_scale();
            match dt {
                DatumType::QU8(_) => {
                    let a = tract_ndarray::Array4::from_shape_fn((1usize, target_h, target_w, 3usize), |(_, _y, _x, c)| {
                        // image::RgbImage index is [x][y]
                        let v = resized[(_x as _, _y as _)][c] as f32;
                        let pixel = v / 255.0_f32;
                        let q = (pixel / scale).round() as i32 + zp;
                        let q = if q < 0 { 0 } else if q > 255 { 255 } else { q };
                        q as u8
                    });
                    let mut t: Tensor = a.into();
                    let desired = DatumType::U8.with_zp_scale(zp, scale);
                    t = t.cast_to_dt(desired)?.into_owned();
                    t
                }
                DatumType::QI8(_) => {
                    let a = tract_ndarray::Array4::from_shape_fn((1usize, target_h, target_w, 3usize), |(_, _y, _x, c)| {
                        let v = resized[(_x as _, _y as _)][c] as f32;
                        let pixel = v / 255.0_f32;
                        let q = (pixel / scale).round() as i32 + zp;
                        let q = if q < -128 { -128 } else if q > 127 { 127 } else { q };
                        q as i8
                    });
                    let mut t: Tensor = a.into();
                    let desired = DatumType::I8.with_zp_scale(zp, scale);
                    t = t.cast_to_dt(desired)?.into_owned();
                    t
                }
                DatumType::QI32(_) => {
                    let a = tract_ndarray::Array4::from_shape_fn((1usize, target_h, target_w, 3usize), |(_, _y, _x, c)| {
                        let v = resized[(_x as _, _y as _)][c] as f32;
                        let pixel = v / 255.0_f32;
                        let q = (pixel / scale).round() as i32 + zp;
                        q
                    });
                    let mut t: Tensor = a.into();
                    let desired = DatumType::I32.with_zp_scale(zp, scale);
                    t = t.cast_to_dt(desired)?.into_owned();
                    t
                }
                _ => {
                    // Fallback to float if we don't know how to materialize the quantized type
                    tract_ndarray::Array4::from_shape_fn((1usize, target_h, target_w, 3usize), |(_, y, x, c)| {
                        let v = resized[(x as _, y as _)][c] as f32;
                        if as_f32 { v / 255.0 } else { v }
                    })
                    .into()
                }
            }
        }
        _ => tract_ndarray::Array4::from_shape_fn((1usize, target_h, target_w, 3usize), |(_, y, x, c)| {
            let v = resized[(x as _, y as _)][c] as f32;
            if as_f32 { v / 255.0 } else { v }
        })
        .into(),
    };

    // Now optimize and make runnable, then run the model on the input
    // Optional debug: dump intermediate layers if DUMP_LAYER env var is set
    if std::env::var("DUMP_LAYER").is_ok() {
        // try to re-translate a fresh model for debug to avoid consuming `model`
    let debug_model = t.model_for_proto_model(&proto)?;
        // Dump up to N nodes with small outputs (<= MAX_ELEMS) for comparison
        let max_elems: usize = 1000;
        let mut dumped = 0usize;
    // Also prepare targeted names to always dump
    let target_names = ["mean", "fully_connected", "fully_connected1", "softmax", "StatefulPartitionedCall"];
        for (idx, node) in debug_model.nodes().iter().enumerate() {
            if dumped >= 12 { break; }
            // skip source nodes
            if node.inputs.is_empty() { continue; }
            // try to get output fact shape
            if node.outputs.is_empty() { continue; }
            let fact = &node.outputs[0].fact;
            let shape = if let Some(s) = fact.shape.as_concrete() {
                s.iter().map(|d| *d as usize).collect::<Vec<_>>()
            } else { continue };
            let elems: usize = shape.iter().product();
            if elems == 0 || elems > max_elems { continue; }
            let mut dm = debug_model.clone();
            dm.set_output_outlets(&[tract_core::internal::OutletId::new(idx, 0)])?;
            let runnable = tract_core::model::TypedRunnableModel::new(&dm)?;
            let outputs = runnable.run(tvec!(image_tensor.clone().into_tvalue()))?;
            let t = &outputs[0];
            let dt = t.datum_type();
            eprintln!("Dump node {}: name='{}' dtype={:?} shape={:?}", idx, node.name, dt, t.shape());
            if dt.is_float() {
                let arr = t.to_array_view::<f32>()?;
                let min = arr.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = arr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = arr.iter().cloned().sum();
                let mean = sum / arr.len() as f32;
                eprintln!(" stats float min={} max={} mean={}", min, max, mean);
            } else {
                eprintln!(" stats: non-f32 tensor, sample first 10 elements");
                // If quantized, also compute dequantized float stats for comparison
                if dt.is_quantized() {
                    let (zp, scale) = dt.zp_scale();
                    if dt.unquantized() == u8::datum_type() {
                        let s = t.to_array_view::<u8>()?;
                        let slice = &s.as_slice().unwrap()[..s.len().min(10)];
                        eprintln!("ints {:?}", slice);
                        let mut min = f32::INFINITY;
                        let mut max = f32::NEG_INFINITY;
                        let mut sum = 0f64;
                        for v in s.as_slice().unwrap().iter() {
                            let f = ((*v as i32) - zp) as f32 * scale;
                            min = min.min(f);
                            max = max.max(f);
                            sum += f as f64;
                        }
                        let mean = sum / (s.len() as f64);
                        eprintln!(" dequantized stats float min={} max={} mean={}", min, max, mean);
                    } else if dt.unquantized() == i8::datum_type() {
                        let s = t.to_array_view::<i8>()?;
                        let slice = &s.as_slice().unwrap()[..s.len().min(10)];
                        eprintln!("ints {:?}", slice);
                        let mut min = f32::INFINITY;
                        let mut max = f32::NEG_INFINITY;
                        let mut sum = 0f64;
                        for v in s.as_slice().unwrap().iter() {
                            let f = ((*v as i32) - zp) as f32 * scale;
                            min = min.min(f);
                            max = max.max(f);
                            sum += f as f64;
                        }
                        let mean = sum / (s.len() as f64);
                        eprintln!(" dequantized stats float min={} max={} mean={}", min, max, mean);
                    } else if dt.unquantized() == i32::datum_type() {
                        let s = t.to_array_view::<i32>()?;
                        let slice = &s.as_slice().unwrap()[..s.len().min(10)];
                        eprintln!("ints {:?}", slice);
                        let mut min = f32::INFINITY;
                        let mut max = f32::NEG_INFINITY;
                        let mut sum = 0f64;
                        for v in s.as_slice().unwrap().iter() {
                            let f = ((*v as i32) - zp) as f32 * scale;
                            min = min.min(f);
                            max = max.max(f);
                            sum += f as f64;
                        }
                        let mean = sum / (s.len() as f64);
                        eprintln!(" dequantized stats float min={} max={} mean={}", min, max, mean);
                    } else {
                        eprintln!(" raw (unknown int type)");
                    }
                } else {
                    if dt.unquantized() == u8::datum_type() {
                        let s = t.to_array_view::<u8>()?;
                        eprintln!("{:?}", &s.as_slice().unwrap()[..s.len().min(10)]);
                    } else if dt.unquantized() == i8::datum_type() {
                        let s = t.to_array_view::<i8>()?;
                        eprintln!("{:?}", &s.as_slice().unwrap()[..s.len().min(10)]);
                    } else if dt.unquantized() == i32::datum_type() {
                        let s = t.to_array_view::<i32>()?;
                        eprintln!("{:?}", &s.as_slice().unwrap()[..s.len().min(10)]);
                    }
                }
            }
            dumped += 1;
        }
        // Dump targeted nodes by name even if they exceed max_elems
        for (idx, node) in debug_model.nodes().iter().enumerate() {
            if target_names.iter().any(|s| node.name.contains(s)) {
                let mut dm = debug_model.clone();
                dm.set_output_outlets(&[tract_core::internal::OutletId::new(idx, 0)])?;
                let runnable = tract_core::model::TypedRunnableModel::new(&dm)?;
                let outputs = runnable.run(tvec!(image_tensor.clone().into_tvalue()))?;
                let t = &outputs[0];
                let dt = t.datum_type();
                eprintln!("Target Dump node {}: name='{}' dtype={:?} shape={:?}", idx, node.name, dt, t.shape());
                if dt.is_float() {
                    let arr = t.to_array_view::<f32>()?;
                    let min = arr.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = arr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let sum: f32 = arr.iter().cloned().sum();
                    let mean = sum / arr.len() as f32;
                    eprintln!(" stats float min={} max={} mean={}", min, max, mean);
                } else if dt.is_quantized() {
                    let (zp, scale) = dt.zp_scale();
                    if dt.unquantized() == i32::datum_type() {
                        let s = t.to_array_view::<i32>()?;
                        let slice = &s.as_slice().unwrap()[..s.len().min(10)];
                        eprintln!("ints {:?}", slice);
                        let mut min = f32::INFINITY;
                        let mut max = f32::NEG_INFINITY;
                        let mut sum = 0f64;
                        for v in s.as_slice().unwrap().iter() {
                            let f = ((*v as i32) - zp) as f32 * scale;
                            min = min.min(f);
                            max = max.max(f);
                            sum += f as f64;
                        }
                        let mean = sum / (s.len() as f64);
                        eprintln!(" dequantized stats float min={} max={} mean={}", min, max, mean);
                    } else if dt.unquantized() == u8::datum_type() {
                        let s = t.to_array_view::<u8>()?;
                        let slice = &s.as_slice().unwrap()[..s.len().min(10)];
                        eprintln!("ints {:?}", slice);
                        let mut min = f32::INFINITY;
                        let mut max = f32::NEG_INFINITY;
                        let mut sum = 0f64;
                        for v in s.as_slice().unwrap().iter() {
                            let f = ((*v as i32) - zp) as f32 * scale;
                            min = min.min(f);
                            max = max.max(f);
                            sum += f as f64;
                        }
                        let mean = sum / (s.len() as f64);
                        eprintln!(" dequantized stats float min={} max={} mean={}", min, max, mean);
                    }
                }
            }
        }
    }

    // Optional: compare per-node outputs with an unquantized sibling model
    if std::env::var("COMPARE_LAYER").is_ok() {
        // Try to find an unquantized sibling by replacing "model.tflite" with "model_unquant.tflite"
        let other_path = if path.ends_with("model.tflite") {
            path.replace("model.tflite", "model_unquant.tflite")
        } else {
            // not sure where sibling is; try adding _unquant before extension
            if let Some(pos) = path.rfind('.') {
                format!("{}{}_unquant{}", &path[..pos], &path[pos..pos+1], &path[pos..])
            } else { path.clone() }
        };
        if std::path::Path::new(&other_path).exists() {
            eprintln!("COMPARE_LAYER: found sibling model at {}", other_path);
            let mut f2 = std::fs::File::open(&other_path)?;
            let proto2 = t.proto_model_for_read(&mut f2).context("Reading proto model 2")?;
            let model2 = t.model_for_proto_model(&proto2)?;
            // Build per-model input tensors so each model gets the dtype it expects
            let make_input_for = |m: &tract_core::model::TypedModel| -> TractResult<Tensor> {
                // infer shape and dtype
                let mut th = 224usize; let mut tw = 224usize;
                let mut as_f32 = true;
                let mut input_dt: Option<DatumType> = None;
                if let Ok(fact) = m.input_fact(0) {
                    if let Some(shape) = fact.shape.as_concrete() {
                        let dims: TVec<usize> = shape.into();
                        if dims.len() == 4 {
                            if dims[3] == 3 { th = dims[1]; tw = dims[2]; }
                            else if dims[1] == 3 { th = dims[2]; tw = dims[3]; }
                        }
                    }
                    as_f32 = fact.datum_type.is_float();
                    input_dt = Some(fact.datum_type.clone());
                }
                let t: Tensor = match input_dt {
                    Some(dt) if dt.is_quantized() => {
                        let (zp, scale) = dt.zp_scale();
                        match dt {
                            DatumType::QU8(_) => {
                                let a = tract_ndarray::Array4::from_shape_fn((1usize, th, tw, 3usize), |(_, _y, _x, c)| {
                                    let v = resized[(_x as _, _y as _)][c] as f32;
                                    let pixel = v / 255.0_f32;
                                    let q = (pixel / scale).round() as i32 + zp;
                                    let q = if q < 0 { 0 } else if q > 255 { 255 } else { q };
                                    q as u8
                                });
                                let mut t: Tensor = a.into();
                                let desired = DatumType::U8.with_zp_scale(zp, scale);
                                t = t.cast_to_dt(desired)?.into_owned();
                                t
                            }
                            DatumType::QI8(_) => {
                                let a = tract_ndarray::Array4::from_shape_fn((1usize, th, tw, 3usize), |(_, _y, _x, c)| {
                                    let v = resized[(_x as _, _y as _)][c] as f32;
                                    let pixel = v / 255.0_f32;
                                    let q = (pixel / scale).round() as i32 + zp;
                                    let q = if q < -128 { -128 } else if q > 127 { 127 } else { q };
                                    q as i8
                                });
                                let mut t: Tensor = a.into();
                                let desired = DatumType::I8.with_zp_scale(zp, scale);
                                t = t.cast_to_dt(desired)?.into_owned();
                                t
                            }
                            DatumType::QI32(_) => {
                                let a = tract_ndarray::Array4::from_shape_fn((1usize, th, tw, 3usize), |(_, _y, _x, _c)| {
                                    let v = resized[(_x as _, _y as _)][0] as f32;
                                    let pixel = v / 255.0_f32;
                                    let q = (pixel / scale).round() as i32 + zp;
                                    q
                                });
                                let mut t: Tensor = a.into();
                                let desired = DatumType::I32.with_zp_scale(zp, scale);
                                t = t.cast_to_dt(desired)?.into_owned();
                                t
                            }
                            _ => {
                                tract_ndarray::Array4::from_shape_fn((1usize, th, tw, 3usize), |(_, y, x, c)| {
                                    let v = resized[(x as _, y as _)][c] as f32;
                                    if as_f32 { v / 255.0 } else { v }
                                })
                                .into()
                            }
                        }
                    }
                    _ => tract_ndarray::Array4::from_shape_fn((1usize, th, tw, 3usize), |(_, y, x, c)| {
                        let v = resized[(x as _, y as _)][c] as f32;
                        if as_f32 { v / 255.0 } else { v }
                    })
                    .into(),
                };
                Ok(t)
            };

            let image_tensor1 = make_input_for(&model)?;
            let image_tensor2 = make_input_for(&model2)?;

            // For each node in optimized model, if small, dump and compare with same-named node in model2
            let max_elems: usize = 1000;
            for (idx, node) in model.nodes().iter().enumerate() {
                if node.outputs.is_empty() { continue; }
                let fact = &node.outputs[0].fact;
                let shape = if let Some(s) = fact.shape.as_concrete() { s.iter().map(|d| *d as usize).collect::<Vec<_>>() } else { continue };
                let elems: usize = shape.iter().product();
                if elems == 0 || elems > max_elems { continue; }
                // find node with same name in model2
                if let Some((idx2, _)) = model2.nodes().iter().enumerate().find(|(_,n)| n.name == node.name) {
                    // build runnable for each node output
                    let mut m1 = model.clone();
                    m1.set_output_outlets(&[tract_core::internal::OutletId::new(idx, 0)])?;
                    let mut m2 = model2.clone();
                    m2.set_output_outlets(&[tract_core::internal::OutletId::new(idx2, 0)])?;
                    let r1 = tract_core::model::TypedRunnableModel::new(&m1)?;
                    let r2 = tract_core::model::TypedRunnableModel::new(&m2)?;
                    let out1 = r1.run(tvec!(image_tensor1.clone().into_tvalue()))?[0].clone();
                    let out2 = r2.run(tvec!(image_tensor2.clone().into_tvalue()))?[0].clone();
                    // dequantize or convert both to f32 arrays
                    // Prefer proto-specified quantization parameters (including per-axis) when available.
                    fn to_f32(t: &Tensor, proto: Option<&TfliteProtoModel>, tensor_name: &str) -> TractResult<tract_ndarray::ArrayD<f32>> {
                        use tract_tflite::tflite;

                        // Helper: try to find quant params in the proto for a tensor named `tensor_name`.
                        let proto_qp = if let Some(p) = proto {
                            let root = p.root();
                            if let Some(subs) = root.subgraphs() {
                                let main = subs.get(0);
                                if let Some(tensors) = main.tensors() {
                                    let mut found: Option<tflite::Tensor<'_>> = None;
                                    for i in 0..tensors.len() {
                                        let ft = tensors.get(i);
                                        if let Some(n) = ft.name() {
                                            if n == tensor_name {
                                                found = Some(ft);
                                                break;
                                            }
                                        }
                                    }
                                    if let Some(ft) = found {
                                        if let Some(qp) = ft.quantization() {
                                            let scales = qp.scale().map(|s| s.iter().collect::<Vec<f32>>());
                                            let zps = qp.zero_point().map(|z| z.iter().collect::<Vec<i64>>());
                                            let qdim = qp.quantized_dimension();
                                            Some((scales, zps, qdim))
                                        } else { None }
                                    } else { None }
                                } else { None }
                            } else { None }
                        } else { None };

                        // If proto has per-axis or per-tensor data, use it to dequantize element-wise.
                        if let Some((scales_opt, zps_opt, qdim)) = proto_qp {
                            if let (Some(scales), Some(zps)) = (scales_opt, zps_opt) {
                                if scales.len() == 1 && zps.len() == 1 {
                                    // Scalar quantization
                                    let scale = scales[0];
                                    let zp = zps[0] as i32;
                                    if t.datum_type().unquantized() == u8::datum_type() {
                                        let s = t.to_array_view::<u8>()?;
                                        let v = s.as_slice().unwrap().iter().map(|x| (((*x as i32) - zp) as f32) * scale).collect::<Vec<_>>();
                                        return Ok(tract_ndarray::ArrayD::from_shape_vec(s.shape().to_vec(), v)?.into_dyn());
                                    } else if t.datum_type().unquantized() == i8::datum_type() {
                                        let s = t.to_array_view::<i8>()?;
                                        let v = s.as_slice().unwrap().iter().map(|x| (((*x as i32) - zp) as f32) * scale).collect::<Vec<_>>();
                                        return Ok(tract_ndarray::ArrayD::from_shape_vec(s.shape().to_vec(), v)?.into_dyn());
                                    } else if t.datum_type().unquantized() == i32::datum_type() {
                                        let s = t.to_array_view::<i32>()?;
                                        let v = s.as_slice().unwrap().iter().map(|x| (((*x as i32) - zp) as f32) * scale).collect::<Vec<_>>();
                                        return Ok(tract_ndarray::ArrayD::from_shape_vec(s.shape().to_vec(), v)?.into_dyn());
                                    }
                                } else {
                                    // Per-axis quantization: iterate with indices and pick scale/zp based on quantized_dimension
                                    let qdim_usize: usize = if qdim >= 0 { qdim as usize } else { 0 };
                                    if t.datum_type().unquantized() == u8::datum_type() {
                                        let a = t.to_array_view::<u8>()?;
                                        let mut out = tract_ndarray::ArrayD::<f32>::zeros(a.shape());
                                        for (idx, v) in a.indexed_iter() {
                                            let axis_idx = idx[qdim_usize];
                                            let scale = scales[axis_idx];
                                            let zp = zps[axis_idx] as i32;
                                            out[idx] = (((*v as i32) - zp) as f32) * scale;
                                        }
                                        return Ok(out);
                                    } else if t.datum_type().unquantized() == i8::datum_type() {
                                        let a = t.to_array_view::<i8>()?;
                                        let mut out = tract_ndarray::ArrayD::<f32>::zeros(a.shape());
                                        for (idx, v) in a.indexed_iter() {
                                            let axis_idx = idx[qdim_usize];
                                            let scale = scales[axis_idx];
                                            let zp = zps[axis_idx] as i32;
                                            out[idx] = (((*v as i32) - zp) as f32) * scale;
                                        }
                                        return Ok(out);
                                    } else if t.datum_type().unquantized() == i32::datum_type() {
                                        let a = t.to_array_view::<i32>()?;
                                        let mut out = tract_ndarray::ArrayD::<f32>::zeros(a.shape());
                                        for (idx, v) in a.indexed_iter() {
                                            let axis_idx = idx[qdim_usize];
                                            let scale = scales[axis_idx];
                                            let zp = zps[axis_idx] as i32;
                                            out[idx] = (((*v as i32) - zp) as f32) * scale;
                                        }
                                        return Ok(out);
                                    }
                                }
                            }
                        }

                        // Special-case fallback: some quantized models store bias as INT32 buffers
                        // without explicit quantization fields on the tensor. Try to detect that
                        // in the proto: if the proto tensor type is INT32 and the runtime tensor
                        // is currently a float (imported incorrectly), read the raw buffer from
                        // the proto, interpret as i32 little-endian and dequantize using the
                        // product of input_scale * kernel_scale found on the consuming operator.
                        if let Some(p) = proto {
                            let root = p.root();
                            if let Some(subs) = root.subgraphs() {
                                let main = subs.get(0);
                                if let Some(tensors) = main.tensors() {
                                    // find proto tensor index by name
                                    for ti in 0..tensors.len() {
                                        let ft = tensors.get(ti);
                                        if ft.name().map(|n| n == tensor_name).unwrap_or(false) {
                                            // proto tensor found
                                            if ft.type_() == tflite::TensorType::INT32 {
                                                // try to locate an operator that consumes this tensor
                                                if let Some(ops) = main.operators() {
                                                    for op_i in 0..ops.len() {
                                                        let op = ops.get(op_i);
                                                        if let Some(inputs) = op.inputs() {
                                                            // if this tensor index appears as one of op inputs
                                                            if inputs.iter().any(|x| x as usize == ti) {
                                                                // attempt to get input and kernel quant scales
                                                                if inputs.len() >= 2 {
                                                                    let in_idx = inputs.get(0) as usize;
                                                                    let k_idx = inputs.get(1) as usize;
                                                                    if let Some(tensors_vec) = main.tensors() {
                                                                        if in_idx < tensors_vec.len() && k_idx < tensors_vec.len() {
                                                                            let in_t = tensors_vec.get(in_idx);
                                                                            let k_t = tensors_vec.get(k_idx);
                                                                            let in_scale_opt = in_t.quantization().and_then(|qp| qp.scale()).map(|s| s.get(0));
                                                                            let k_scale_opt = k_t.quantization().and_then(|qp| qp.scale()).map(|s| s.get(0));
                                                                            if let (Some(in_scale), Some(k_scale)) = (in_scale_opt, k_scale_opt) {
                                                                                // read buffer bytes and reinterpret as i32 LE
                                                                                if let Some(buffers) = root.buffers() {
                                                                                    let buf_ix = ft.buffer() as usize;
                                                                                    if buf_ix < buffers.len() {
                                                                                        let b = buffers.get(buf_ix);
                                                                                        if let Some(data) = b.data() {
                                                                                            let bytes = data.bytes();
                                                                                            use std::convert::TryInto;
                                                                                            let n_elems = bytes.len() / 4;
                                                                                            let mut v = Vec::with_capacity(n_elems);
                                                                                            for i in 0..n_elems {
                                                                                                let off = i * 4;
                                                                                                if off + 4 > bytes.len() { break; }
                                                                                                let w: [u8;4] = bytes[off..off+4].try_into().unwrap();
                                                                                                let vi = i32::from_le_bytes(w);
                                                                                                v.push((vi as f32) * in_scale * k_scale);
                                                                                            }
                                                                                            return Ok(tract_ndarray::ArrayD::from_shape_vec(vec![n_elems], v)?);
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Fallback: use datum_type's zp/scale if tensor is quantized
                        if t.datum_type().is_float() {
                            return Ok(t.to_array_view::<f32>()?.to_owned().into_dyn());
                        }
                        if t.datum_type().is_quantized() {
                            let (zp, scale) = t.datum_type().zp_scale();
                            if t.datum_type().unquantized() == u8::datum_type() {
                                let s = t.to_array_view::<u8>()?;
                                let v = s.as_slice().unwrap().iter().map(|x| (((*x as i32) - zp) as f32) * scale).collect::<Vec<_>>();
                                return Ok(tract_ndarray::ArrayD::from_shape_vec(s.shape().to_vec(), v)?.into_dyn());
                            } else if t.datum_type().unquantized() == i8::datum_type() {
                                let s = t.to_array_view::<i8>()?;
                                let v = s.as_slice().unwrap().iter().map(|x| (((*x as i32) - zp) as f32) * scale).collect::<Vec<_>>();
                                return Ok(tract_ndarray::ArrayD::from_shape_vec(s.shape().to_vec(), v)?.into_dyn());
                            } else if t.datum_type().unquantized() == i32::datum_type() {
                                let s = t.to_array_view::<i32>()?;
                                let v = s.as_slice().unwrap().iter().map(|x| (((*x as i32) - zp) as f32) * scale).collect::<Vec<_>>();
                                return Ok(tract_ndarray::ArrayD::from_shape_vec(s.shape().to_vec(), v)?.into_dyn());
                            } else {
                                return Err(format_err!("Unsupported quantized type for comparison {:?}", t.datum_type()));
                            }
                        }
                        // non-quantized integers? convert to f32
                        if t.datum_type().unquantized() == u8::datum_type() {
                            let s = t.to_array_view::<u8>()?;
                            let v = s.as_slice().unwrap().iter().map(|x| *x as f32).collect::<Vec<_>>();
                            return Ok(tract_ndarray::ArrayD::from_shape_vec(s.shape().to_vec(), v)?.into_dyn());
                        } else if t.datum_type().unquantized() == i8::datum_type() {
                            let s = t.to_array_view::<i8>()?;
                            let v = s.as_slice().unwrap().iter().map(|x| *x as f32).collect::<Vec<_>>();
                            return Ok(tract_ndarray::ArrayD::from_shape_vec(s.shape().to_vec(), v)?.into_dyn());
                        } else if t.datum_type().unquantized() == i32::datum_type() {
                            let s = t.to_array_view::<i32>()?;
                            let v = s.as_slice().unwrap().iter().map(|x| *x as f32).collect::<Vec<_>>();
                            return Ok(tract_ndarray::ArrayD::from_shape_vec(s.shape().to_vec(), v)?.into_dyn());
                        }
                        Err(format_err!("Unsupported type for comparison {:?}", t.datum_type()))
                    }
                    // Try to find a better proto tensor name hint: prefer node.name, else try first input node name(s).
                    let mut proto_name1 = node.name.clone();
                    if let Some(p) = Some(&proto) {
                        // if proto does not contain a tensor named node.name, try the first input node name
                        let root = proto.root();
                        let has = |n: &str| {
                            if let Some(subs) = root.subgraphs() {
                                let main = subs.get(0);
                                if let Some(tensors) = main.tensors() {
                                    for i in 0..tensors.len() {
                                        if let Some(tn) = tensors.get(i).name() {
                                            if tn == n { return true; }
                                        }
                                    }
                                }
                            }
                            false
                        };
                        if !has(&proto_name1) {
                            if let Some(first_input) = node.inputs.get(0) {
                                let in_node = &model.nodes()[first_input.node];
                                proto_name1 = in_node.name.clone();
                            }
                        }
                    }
                    let mut proto_name2 = node.name.clone();
                    if let Some(p2) = Some(&proto2) {
                        let root2 = proto2.root();
                        let has2 = |n: &str| {
                            if let Some(subs) = root2.subgraphs() {
                                let main = subs.get(0);
                                if let Some(tensors) = main.tensors() {
                                    for i in 0..tensors.len() {
                                        if let Some(tn) = tensors.get(i).name() {
                                            if tn == n { return true; }
                                        }
                                    }
                                }
                            }
                            false
                        };
                        if !has2(&proto_name2) {
                            if let Some(first_input) = node.inputs.get(0) {
                                let in_node = &model2.nodes()[first_input.node];
                                proto_name2 = in_node.name.clone();
                            }
                        }
                    }
                    let a1 = to_f32(&out1, Some(&proto), &proto_name1)?;
                    let a2 = to_f32(&out2, Some(&proto2), &proto_name2)?;
                    if a1.shape() != a2.shape() { eprintln!("COMPARE mismatch shapes for node {}: {:?} vs {:?}", node.name, a1.shape(), a2.shape()); continue; }
                    // compute L2 and max abs diff
                    let mut l2 = 0f64; let mut max = 0f32;
                    for (x, y) in a1.iter().zip(a2.iter()) {
                        let d = *x - *y;
                        l2 += (d as f64) * (d as f64);
                        max = max.max(d.abs());
                    }
                    l2 = (l2 / (a1.len() as f64)).sqrt();
                    eprintln!("COMPARE node {} idx1={} idx2={} elems={} l2_rms={} max_abs={}", node.name, idx, idx2, a1.len(), l2, max);
                    // If we see NaN or a large max difference, dump more info to help debug
                    if l2.is_nan() || max > 1e-2 {
                        eprintln!("Large divergence at node {}: dumping arrays for deeper inspection", node.name);
                        eprintln!(" proto names: '{}' vs '{}'", proto_name1, proto_name2);
                        eprintln!(" a1 shape={:?} a2 shape={:?}", a1.shape(), a2.shape());
                        // show first 32 elements or fewer
                        let show_n = a1.len().min(32);
                        eprintln!(" a1 first {}: {:?}", show_n, &a1.as_slice().unwrap()[..show_n]);
                        eprintln!(" a2 first {}: {:?}", show_n, &a2.as_slice().unwrap()[..show_n]);
                        // if NaNs present, show their indices
                        let mut nan_indices = vec![];
                        for (i, (x, y)) in a1.iter().zip(a2.iter()).enumerate() {
                            if x.is_nan() || y.is_nan() { nan_indices.push(i); if nan_indices.len() >= 8 { break; } }
                        }
                        if !nan_indices.is_empty() { eprintln!(" NaN indices (up to 8): {:?}", nan_indices); }
                        eprintln!("Stopping further comparison");
                        break;
                    }
                }
            }
        } else {
            eprintln!("COMPARE_LAYER: sibling model not found at {}", other_path);
        }
        // continue to regular run after comparison
    }

    let runnable = model.into_optimized()?.into_runnable()?;
    let result = runnable.run(tvec!(image_tensor.into()))?;

    // Debug: print raw output dtype and small samples to understand quantization
    eprintln!("Raw model output datum_type={:?}", result[0].datum_type());
    if result[0].datum_type().is_quantized() {
        let dt = result[0].datum_type();
        eprintln!("Output qparams: zp={}, scale={}", dt.zp_scale().0, dt.zp_scale().1);
        if dt.unquantized() == u8::datum_type() {
            let s = result[0].to_array_view::<u8>()?;
            eprintln!("Output raw ints sample: {:?}", &s.as_slice().unwrap()[..s.len().min(10)]);
            let (zp, scale) = dt.zp_scale();
            let mut min = f32::INFINITY; let mut max = f32::NEG_INFINITY; let mut sum = 0f64;
            for v in s.as_slice().unwrap().iter() {
                let f = ((*v as i32) - zp) as f32 * scale;
                min = min.min(f); max = max.max(f); sum += f as f64;
            }
            eprintln!("Output dequantized stats float min={} max={} mean={}", min, max, sum / (s.len() as f64));
        } else if dt.unquantized() == i32::datum_type() {
            let s = result[0].to_array_view::<i32>()?;
            eprintln!("Output raw ints sample: {:?}", &s.as_slice().unwrap()[..s.len().min(10)]);
            let (zp, scale) = dt.zp_scale();
            let mut min = f32::INFINITY; let mut max = f32::NEG_INFINITY; let mut sum = 0f64;
            for v in s.as_slice().unwrap().iter() {
                let f = ((*v as i32) - zp) as f32 * scale;
                min = min.min(f); max = max.max(f); sum += f as f64;
            }
            eprintln!("Output dequantized stats float min={} max={} mean={}", min, max, sum / (s.len() as f64));
        }
    }

    // Ensure output is float for printing top-1; cast quantized outputs to f32.
    let output_tensor: Tensor = if result[0].datum_type().is_quantized() {
        result[0].cast_to::<f32>()?.into_owned()
    } else {
        result[0].clone().into_tensor()
    };

    // Print top-1 if float outputs
    let best = output_tensor
        .to_array_view::<f32>()?
        .iter()
        .cloned()
        .zip(1..)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    println!("result: {best:?}");
    Ok(())
}