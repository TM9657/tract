use tract_core::internal::TractErrorContext;
use tract_core::internal::{format_err, tract_ndarray};
use tract_tflite::tflite;
use tract_tflite::Tflite;
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
        // decide whether to normalize to 0..1 depending on datum type
        use tract_core::prelude::DatumType;
        as_f32 = match fact.datum_type {
            DatumType::U8 => false,
            _ => true,
        };
    }

    let resized = image::imageops::resize(&img, target_w as u32, target_h as u32, ::image::imageops::FilterType::Triangle);
    let image_tensor: Tensor = tract_ndarray::Array4::from_shape_fn((1usize, target_h, target_w, 3usize), |(_, y, x, c)| {
        let v = resized[(x as _, y as _)][c] as f32;
        if as_f32 { v / 255.0 } else { v }
    })
    .into();

    // Now optimize and make runnable, then run the model on the input
    let runnable = model.into_optimized()?.into_runnable()?;
    let result = runnable.run(tvec!(image_tensor.into()))?;

    // Print top-1 if float outputs
    let best = result[0]
        .to_array_view::<f32>()?
        .iter()
        .cloned()
        .zip(1..)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    println!("result: {best:?}");
    Ok(())
}