============================================
ML PIPELINE TEMPLATE
============================================

STEP 1: LOAD DATA
- Read CSV/files into memory
- Example: train_df = pd.read_csv('train.csv')

STEP 2: SEPARATE FEATURES AND LABELS
- X = input data (pixels, measurements, etc.)
- y = what you're trying to predict (labels)
- Example: X = data.drop('label', axis=1), y = data['label']

STEP 3: RESHAPE (if needed)
- Match the shape your model expects
- For CNN: (batch, channels, height, width)
- Example: X = X.reshape(-1, 1, 28, 28)

STEP 4: NORMALIZE
- Scale inputs to small values (0-1 or -1 to 1)
- Example: X = X / 255.0

STEP 5: TRAIN/VALIDATION SPLIT
- Hold out some data to check for overfitting
- Example: 80% train, 20% validation

STEP 6: CREATE DATALOADERS
- Batch your data for efficient training
- Example: DataLoader(dataset, batch_size=64)

STEP 7: BUILD MODEL
- Define architecture (CNN, MLP, etc.)

STEP 8: DEFINE LOSS AND OPTIMIZER
- Loss: how wrong is the model? (CrossEntropyLoss for classification)
- Optimizer: how to update weights (Adam is common)

STEP 9: TRAINING LOOP
- For each epoch:
    - Forward pass: predictions = model(X)
    - Compute loss: loss = criterion(predictions, y)
    - Backward pass: loss.backward()
    - Update weights: optimizer.step()
- Check validation accuracy after each epoch

STEP 10: PREDICT ON TEST SET
- model.eval()
- with torch.no_grad(): predictions = model(X_test)

STEP 11: FORMAT AND SUBMIT
- Create submission file in required format
- Submit to Kaggle