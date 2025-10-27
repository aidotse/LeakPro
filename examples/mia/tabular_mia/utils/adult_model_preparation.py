import pickle

from torch import cuda, device, nn, no_grad, optim, save, sigmoid
from tqdm import tqdm

from leakpro.leakpro import LeakPro
from leakpro.schemas import EvalOutput, TrainingOutput


class AdultNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AdultNet, self).__init__()
        self.init_params = {"input_size": input_size,
                            "hidden_size": hidden_size,
                            "num_classes": num_classes}
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

def evaluate(model, loader, criterion, device):
    model.eval()
    loss, acc = 0, 0
    with no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            target = target.float().unsqueeze(1)
            output = model(data)
            loss += criterion(output, target).item()
            pred = sigmoid(output) >= 0.5
            acc += pred.eq(target.data.view_as(pred)).sum()
        loss /= len(loader)
        acc = float(acc) / len(loader.dataset)
    return loss, acc

def create_trained_model_and_metadata(model, train_loader, test_loader, epochs = 10, metadata = None):

    device_name = device("cuda" if cuda.is_available() else "cpu")
    model.to(device_name)
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8)
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for e in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        train_acc, train_loss = 0.0, 0.0

        for data, target in train_loader:
            target = target.float().unsqueeze(1)
            data, target = data.to(device_name, non_blocking=True), target.to(device_name, non_blocking=True)
            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            pred = output >= 0.5
            train_acc += pred.eq(target).sum().item()

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        test_loss, test_acc = evaluate(model, test_loader, criterion, device_name)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

    # Move the model back to the CPU
    model.to("cpu")
    with open("target/target_model.pkl", "wb") as f:
        save(model.state_dict(), f)

    train_metrics = EvalOutput(
        accuracy=float(train_accuracies[-1]),
        loss=float(train_losses[-1]),
        extra={"accuracy_history": train_accuracies, "loss_history": train_losses},
    )
    train_result = TrainingOutput(model=model, metrics=train_metrics)

    test_metrics = EvalOutput(
        accuracy=float(test_accuracies[-1]),
        loss=float(test_losses[-1]),
        extra={"accuracy_history": test_accuracies, "loss_history": test_losses},
    )

    meta_data = LeakPro.make_mia_metadata(
        train_result=train_result,
        optimizer=optimizer,
        loss_fn=criterion,
        dataloader=train_loader,
        test_result=test_metrics,
        epochs=epochs,
        train_indices=list(train_loader.dataset.indices),
        test_indices=list(test_loader.dataset.indices),
        dataset_name="adult",
    )
    
    with open("target/model_metadata.pkl", "wb") as f:
        pickle.dump(meta_data, f)
    
    return train_accuracies, train_losses, test_accuracies, test_losses
