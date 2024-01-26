'''
optimizer.zero_grad()
loss.backward()

#torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

optimizer.step()
'''
#오차역전파와 step사이에 사용?
#torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2.0)과 같이 사용