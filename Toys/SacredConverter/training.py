from mask import *


def run(_model, _optimizer, _criterion, split):
    _model.train() if split == "train" else _model.eval()
    _data_iter = VerseIterator(split=split)
    _dataloader = DataLoader(_data_iter, batch_size=BATCH_SIZE, collate_fn=collator)

    losses = 0

    count = 0
    for source_batch, target_batch in _dataloader:
        source_batch = source_batch.to(DEVICE)
        target_batch = target_batch.to(DEVICE)
        '''
        print('-----------------')
        print(source_batch)
        print('-----------------')
        print(target_batch)
        '''

        target_input = target_batch[:-1, :]
        target_output = target_batch[1:, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            source_batch, target_input
        )

        logits = _model(
            src=source_batch,
            trg=target_input,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

        _optimizer.zero_grad()
        loss = _criterion(logits.reshape(-1, logits.shape[-1]), target_output.reshape(-1))
        if split == "train":
            loss.backward()
            _optimizer.step()
        losses += loss.item()

        count += 1
        print(count * 16)

    if split == "train":
        return losses / 1809  # len(list(_dataloader))
    else:
        return losses / 67


for epoch in range(100):
    train_loss = run(model, optimizer, criterion, "train")
    val_loss = run(model, optimizer, criterion, "valid")
    print(f"Epoch: {epoch + 1}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")
    torch.save(model.state_dict(), f'./models/{epoch}sentence2bible.pt')

# 5
