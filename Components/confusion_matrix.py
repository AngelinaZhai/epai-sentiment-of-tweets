# def get_accuracy(net, loader, label_names):
#     for i, data in enumerate(loader, 0):
#         inputs, labels = data
#         if use_cuda and torch.cuda.is_available():
#           inputs= inputs.cuda()
#           labels = labels.cuda()
#           net = net.cuda()
#         outputs = net(inputs)
#         labels.float()
#         classification_report(
#             labels,
#             outputs,
#             output_dict=False,
#             target_names=label_names
#         )
#     return

#calculate the f1 score for each label
def get_f1_score(net, loader):
    f1_scores = []
    for i, data in enumerate(loader, 0):
        # get expected and predicted labels
        inputs, labels = data
        if use_cuda and torch.cuda.is_available():
            inputs= inputs.cuda()
            labels = labels.cuda()
            net = net.cuda()
        outputs = net(inputs)

        #convert outputs and labels to numpy arrays with discrete values
        cutoff = 0.7
        outputs = outputs.detach().numpy()
        outputs_np = np.zeros_like(outputs)
        outputs_np[outputs > cutoff] = 1 
        # y_pred_classes[y_pred > cutoff] = 1
        # outputs = np.where(outputs > cutoff, 1, 0)
        labels = labels.detach().numpy()
        labels_np = np.zeros_like(labels)
        labels_np[labels > cutoff] = 1
        # labels = np.where(labels > cutoff, 1, 0)

        f1_scores.append(hamming_loss(labels_np, outputs_np)) #actual vs predicted

    print(f1_scores)
    #print length of f1_scores
    # print(len(f1_scores))
    return f1_scores

# def get_confusion_matrix(net: nn.Module, loader: torch.utils.data.DataLoader, label: str):
#      """
#      Returns the relevant confusion matrix corresponding to the input model and label
#      :param net: the model being tested
#      :param loader: the test data
#      :param label: the label being predicted

#      NOTE: the label must be in the list ["sentiment", "respect", "insult",
#              "humiliate", "status", "dehumanize", "violence", "genocide"
#              "attack_defend"]
#      """
#      index = {"sentiment": 0, "respect": 1, "insult": 2, "humiliate": 3, "status": 4,
#               "dehumanize": 5, "violence": 6, "genocide": 7, "attack_defend": 8}

#      actual = []
#      predicted = []
#      relevant_index = index[label]

#      # get actual stuff
#      for _, data in enumerate(loader, 0):
#          inputs, labels = data
#          curr_tensor = torch.FloatTensor(inputs)

#          actual_value = labels[relevant_index]  # how do I get the actual stuff
#          predicted_value = (net.forward(curr_tensor))[relevant_index]

#          # using 0.5 cutoff

#          if actual_value > 0.5:
#              actual.append(1)
#          else:
#              actual.append(0)

#          if predicted_value > 0.5:
#              predicted.append(1)
#          else:
#              predicted.append(0)

#      # make confusion matrix
#      for i in range(len(actual)):
#          actual[i] = pd.Series(actual[i], name=('Actual ' + label))
#          predicted[i] = pd.Series(predicted[i], name=('Predicted ' + label))
#          print(pd.crosstab(actual[i], predicted[i]))
         

    