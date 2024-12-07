from cond_integrate import *
from temp_hparams import hparams, rel2abs

# model load, state_dict에서 필요한 것만 가져온다
cond_emb_model = ConditionEmbedding(hparams)
state_dict = load_cond_embedding_state(rel2abs(hparams['emb_model_path']))
cond_emb_model.load_state_dict(state_dict)
cond_emb_model.to(hparams['device'])
# print(cond_emb_model)
# for name, param in cond_emb_model.named_parameters():
#     print(name, param.data)

sample = get_tensor_cond(get_raw_cond(*load_cond_model(hparams), hparams, rel2abs(hparams['raw_wave_path'])), hparams)
get_collated_cond(sample)

cond_emb_model.eval()
print(cond_emb_model(sample)['decoder_inp'].shape)
print(cond_emb_model(sample)['f0_denorm'].shape)
