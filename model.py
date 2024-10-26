import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

data = {
    "texts": [
        "This is a scam call",
        "Your account has been compromised",
        "Hello, how can I help you?",
        "Congratulations, you have won a prize",
        "Please provide your bank details",
        "This is a legitimate call from your bank",
        "You have an outstanding payment",
        "Your order has been shipped",
        "We need to verify your account information",
        "Thank you for your purchase"
    ],
    "labels": [1, 1, 0, 1, 1, 0, 1, 0, 1, 0]
}

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["texts"]).toarray()
y = torch.tensor(data["labels"], dtype=torch.float32)

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), self.y[idx]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
train_dataset = TextDataset(X_train, y_train)
val_dataset = TextDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

class ScamClassifier(nn.Module):
    def __init__(self, input_dim):
        super(ScamClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

model = ScamClassifier(input_dim=X.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss/len(val_loader)}")

model.eval()
example_text = ["This is a scam call"]
example_vector = vectorizer.transform(example_text).toarray()
example_tensor = torch.tensor(example_vector, dtype=torch.float32)
prediction = model(example_tensor).item()
print(f"Prediction: {'Scam' if prediction > 0.5 else 'Not Scam'}")

torch.save(model.state_dict(), 'scam_classifier.pth')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
