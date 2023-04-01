import torch
import net
adaface_models={'ir_18':"adaface_ir18_casia.ckpt"}
#load model onnx
def load_pretrained_model(architecture='ir_18'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model
if __name__=="__main__":
    model=load_pretrained_model()
    output_onnx = 'FaceRecog.onnx'
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["input0"]
    output_names = ["output0"]
    inputs = torch.randn(1, 3,112,112)
    torch_out = torch.onnx.export(model, inputs, output_onnx, export_params=True, verbose=False,
                                    input_names=input_names, output_names=output_names)
