use tract_core::internal::*;
use tract_core::ops::quant::{QuantizeLinearI8, QuantizeLinearU8};
use tract_core::ops::element_wise::ElementWiseOp;

use crate::registry::DeserOp;

pub fn register_all(reg: &mut crate::registry::Registry) {
    reg.reg_to_tract(crate::tflite::BuiltinOperator::QUANTIZE, de_quantize);
}

fn de_quantize(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    // TFLite Quantize maps a float input to a quantized output using the
    // quantization parameters attached to the output tensor. The tflite
    // tensors reader already set the expected output datum_type with qparams.
    let inputs = op.inputs;
    ensure!(inputs.len() == 1, "QUANTIZE expects single input");

    // figure out target dtype and qparams from declared output fact
    let out_dt = op.output_facts[0].datum_type;
    let (zp, scale) = out_dt.zp_scale();

    let prefix = op.prefix;

    let op_box: Box<dyn TypedOp> = match out_dt.unquantized() {
        DatumType::U8 => Box::new(ElementWiseOp(Box::new(QuantizeLinearU8 { scale: scale.recip(), zero_point: zp as u8 }), Some(out_dt))),
        DatumType::I8 => Box::new(ElementWiseOp(Box::new(QuantizeLinearI8 { scale: scale.recip(), zero_point: zp as i8 }), Some(out_dt))),
        _ => bail!("Unsupported quantize target datatype: {:?}", out_dt),
    };

    op.ctx.target.wire_node(prefix, op_box, &[inputs[0]])
}
