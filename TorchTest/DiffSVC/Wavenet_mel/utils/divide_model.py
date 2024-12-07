import torch

epoch = 38100
ckpt_path = f'model_ckpt_steps_{epoch}.ckpt'


def load_cond_embedding_state(model_path):
    model_state_dict = torch.load(model_path, map_location='cpu')['state_dict']
    # print(model_state_dict.keys())
    embedding_state_dict = {
        "mel_out.weight": model_state_dict["model.fs2.mel_out.weight"],
        "mel_out.bias": model_state_dict["model.fs2.mel_out.bias"],
        "pitch_embed.weight": model_state_dict["model.fs2.pitch_embed.weight"],
    }
    diff_state_dict = dict()
    for key in model_state_dict.keys():
        key_split_list = key.split('.')
        if key_split_list[1] == 'denoise_fn':
            new_key = ''
            for item in key_split_list[2:]:
                new_key = new_key + item + '.'
            new_key = new_key[:-1]
            diff_state_dict[new_key] = model_state_dict[key]

    return embedding_state_dict, diff_state_dict


embedding_model_state_dict, wavenet_model_state_dict = load_cond_embedding_state(ckpt_path)
# print(embedding_model_state_dict.keys())
torch.save(embedding_model_state_dict, f'embedding_model_steps_{epoch}.pt')
# print(wavenet_model_state_dict.keys())
torch.save(wavenet_model_state_dict, f'wavenet_model_steps_{epoch}.pt')
