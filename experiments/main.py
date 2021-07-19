
    def configure_optimizers(self):
        optimizer = MADGRAD(self.parameters(), lr=LEARNING_RATE)
        return optimizer

    def focal_tversky_loss(self, x, y):
        loss = FocalTverskyLoss()(x, y)
        return loss

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        outputs, main_loss, aux_loss = self.forward(inputs, labels)

        probs = torch.sigmoid(outputs).data.cpu().numpy()
        crf = np.stack(list(map(dense_crf_wrapper, zip(inputs.cpu().numpy(), probs))))
        crf = np.ascontiguousarray(crf)
        f1_crf = f1_score(torch.from_numpy(crf).to(self.device), labels)

        loss = main_loss + ALPHA_AUX * aux_loss
        f1 = f1_score(outputs, labels)

        self.log('train_loss', loss)
        self.log('train_f1', f1)
        self.log('train_f1_crf', f1_crf)
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        outputs = self.forward(inputs, labels)

        probs = torch.sigmoid(outputs).data.cpu().numpy()
        crf = np.stack(list(map(dense_crf_wrapper, zip(inputs.cpu().numpy(), probs))))
        crf = np.ascontiguousarray(crf)
        f1_crf = f1_score(torch.from_numpy(crf).to(self.device), labels)

        loss = self.focal_tversky_loss(outputs.float(), labels.float())
        f1 = f1_score(outputs, labels)

        self.log('val_loss', loss)
        self.log('val_f1_crf', f1_crf)
        self.log('val_f1', f1)

