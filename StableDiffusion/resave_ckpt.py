import torch,os
device = 'cpu'
model_file1 = "./data/v1-5-pruned.ckpt"
model_file2 = "./data/v1-5-pruned-emaonly.ckpt"


def check(input_model):
    diffusion_dict = {}
    encoder_dict   = {}
    decoder_dict   = {}
    clip_dict      = {}
    other_dict      = {}


    uniquek=[]
    for k, v in input_model.items():
        splitk = k.split(".")
        thisK = ""
        if len(splitk)==1:
            thisK = splitk[0]
        else:
            thisK = splitk[0]+'.'+splitk[1]
        # thisk = k[:splitk[0]]+'.'+ k[splitk[0]:splitk[1]]
        if not thisK in uniquek:
            uniquek.append(thisK)
    
    for k in uniquek:
        print(k)


    for k, v in input_model.items():
        if k.startswith("model.diffusion_model."):
            new_k = k.replace("model.diffusion_model.", "")
            diffusion_dict[new_k] = v

        # ---- VAE encoder (plus quant_conv) ----
        elif k.startswith("first_stage_model.encoder."):
            new_k = k.replace("first_stage_model.encoder.", "")
            encoder_dict[new_k] = v
        elif k.startswith("first_stage_model.quant_conv."):   # matches .weight/.bias
            new_k = k.replace("first_stage_model.quant_conv.", "preConv.")
            encoder_dict[new_k] = v

        # ---- VAE decoder (plus post_quant_conv) ----
        elif k.startswith("first_stage_model.decoder."):
            new_k = k.replace("first_stage_model.decoder.", "")
            decoder_dict[new_k] = v
        elif k.startswith("first_stage_model.post_quant_conv."):  # matches .weight/.bias
            new_k = k.replace("first_stage_model.post_quant_conv.", "postConv.")
            decoder_dict[new_k] = v

        # ---- Text encoder (CLIP) ----
        elif k.startswith("cond_stage_model."):
            new_k = k.replace("cond_stage_model.transformer.", "")
            clip_dict[new_k] = v
        
        # ---- Others ----
        else:
            other_dict[k] = v
    return diffusion_dict,encoder_dict,decoder_dict,clip_dict,other_dict



original_model1 = torch.load(model_file1, map_location=device, weights_only = False)["state_dict"]
original_model2 = torch.load(model_file2, map_location=device, weights_only = False)["state_dict"]


diffusion_dict1,encoder_dict1,decoder_dict1,clip_dict1,other_dict1 = check(original_model1)
diffusion_dict2,encoder_dict2,decoder_dict2,clip_dict2,other_dict2 = check(original_model2)


torch.save({"state_dict": diffusion_dict1}, os.path.join('models', "diffusion.ckpt"))
torch.save({"state_dict": encoder_dict1},   os.path.join('models', "vae_encoder.ckpt"))
torch.save({"state_dict": decoder_dict1},   os.path.join('models', "vae_decoder.ckpt"))
torch.save({"state_dict": clip_dict1},      os.path.join('models', "clip.ckpt"))
torch.save({"state_dict": other_dict1},      os.path.join('models', "others.ckpt"))


torch.save({"state_dict": diffusion_dict2}, os.path.join('models', "diffusion_emaonly.ckpt"))
torch.save({"state_dict": encoder_dict2},   os.path.join('models', "vae_encoder_emaonly.ckpt"))
torch.save({"state_dict": decoder_dict2},   os.path.join('models', "vae_decoder_emaonly.ckpt"))
torch.save({"state_dict": clip_dict2},      os.path.join('models', "clip_emaonly.ckpt"))
torch.save({"state_dict": other_dict2},      os.path.join('models', "others_emaonly.ckpt"))

