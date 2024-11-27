import torch
import glob
import os
import re


def load_ckpt(cur_model, ckpt_base_dir, prefix_in_ckpt='model', force=True, strict=True):
    if os.path.isfile(ckpt_base_dir):
        base_dir = os.path.dirname(ckpt_base_dir)
        checkpoint_path = [ckpt_base_dir]
    else:
        base_dir = ckpt_base_dir
        checkpoint_path = sorted(glob.glob(f'{base_dir}/model_ckpt_steps_*.ckpt'), key=
        lambda x: int(re.findall(f'{base_dir}/model_ckpt_steps_(\d+).ckpt', x.replace('\\', '/'))[0]))
    if len(checkpoint_path) > 0:
        checkpoint_path = checkpoint_path[-1]
        state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        state_dict = {k[len(prefix_in_ckpt) + 1:]: v for k, v in state_dict.items()
                      if k.startswith(f'{prefix_in_ckpt}.')}
        if not strict:
            cur_model_state_dict = cur_model.state_dict()
            unmatched_keys = []
            for key, param in state_dict.items():
                if key in cur_model_state_dict:
                    new_param = cur_model_state_dict[key]
                    if new_param.shape != param.shape:
                        unmatched_keys.append(key)
                        print("| Unmatched keys: ", key, new_param.shape, param.shape)
            for key in unmatched_keys:
                del state_dict[key]

        '''
        원본은 class GaussianDiffusion(nn.Module)을 통째로 저장해놨으므로,
        denoise_fn이 달린 것만 조용히 빼오자
        '''
        diff_state_dict = dict()
        for key in state_dict.keys():
            key_split_list = key.split('.')
            if key_split_list[0] == 'denoise_fn':
                new_key = ''
                for item in key_split_list[1:]:
                    new_key = new_key + item + '.'
                new_key = new_key[:-1]
                diff_state_dict[new_key] = state_dict[key]
        '''for key, value in diff_state_dict.items():
            print(key)  # , value)'''

        cur_model.load_state_dict(diff_state_dict, strict=strict)  # state_dict, strict=strict)
        print(f"| load '{prefix_in_ckpt}' from '{checkpoint_path}'.")
    else:
        e_msg = f"| ckpt not found in {base_dir}."
        if force:
            assert False, e_msg
        else:
            print(e_msg)
