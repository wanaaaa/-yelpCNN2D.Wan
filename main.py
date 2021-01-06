from funClass import *

top_data_df = wordProcess()

X_train, X_test, Y_train, Y_test = split_train_test(top_data_df)

w2vmodel, word2vec_file = make_word2vec_model(top_data_df, padding=True, sg=1, min_count=1,
            size=10, workers=1, window=3)

NUM_CLASSES = 3
VOCAB_SIZE = len(w2vmodel.wv.vocab)

cnn_model = CnnTextClassifier(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES)
cnn_model.to('cpu')
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
num_epochs = 1

loss_file_name = './plots/' + 'cnn_class_big_loss_with_padding.csv'
f = open(loss_file_name, 'w')
f.write('iter, loss')
f.write('\n')
losses = []
cnn_model.train()
max_sen_len = top_data_df.stemmed_tokens.map(len).max()
padding_idx = w2vmodel.wv.vocab['pad'].index
for epoch in range(num_epochs):
    print("Epoch" + str(epoch + 1))
    train_loss = 0
    for index, row in X_train.iterrows():
        # Clearing the accumulated gradients
        cnn_model.zero_grad()

        # Make the bag of words vector for stemmed tokens
        bow_vec = make_word2vec_vector_cnn(row['stemmed_tokens'], w2vmodel,
                                           max_sen_len, padding_idx)

        # Forward pass to get output
        probs = cnn_model(bow_vec)

        # Get the target label
        target = make_target(Y_train['sentiment'][index])
        print("target===>>>", target)
        print("probs===>", probs)
        # Calculate Loss: softmax --> cross entropy loss
        loss = loss_function(probs, target)
        train_loss += loss.item()

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

    # if index == 0:
    #     continue
    print("Epoch ran :" + str(epoch + 1))
    f.write(str((epoch + 1)) + "," + str(train_loss / len(X_train)))
    f.write('\n')
    train_loss = 0

torch.save(cnn_model, 'cnn_big_model_500_with_padding.pth')

f.close()
# print("Input vector")
# print(bow_vec.cpu().numpy())
# print("Probs")
# print(probs)
# print(torch.argmax(probs, dim=1).cpu().numpy()[0])