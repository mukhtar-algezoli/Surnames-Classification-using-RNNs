from SurnameDataset import *
from SurnameClassifier import *
from Dataloader import *
from argparse import Namespace
import torch
import torch.optim as optim
from tqdm import tqdm_notebook

PATH = "savedmodel.tar"

def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100

args = Namespace(
    # Data and Path hyper parameters
    surname_csv="surnames/surnames_with_splits.csv",
    vectorizer_file="vectorizer.json",
    model_state_file="model.pth",
    save_dir="model_storage/ch6/surname_classification",
    # Model hyper parameter
    char_embedding_size=100,
    rnn_hidden_size=64,
    # Training hyper parameter
    num_epochs=100,
    learning_rate=1e-3,
    batch_size=64,
    seed=1337,
    early_stopping_criteria=5,
    # Runtime hyper parameter
    cuda=True,
    catch_keyboard_interrupt=True,
    reload_from_files=False,
    expand_filepaths_to_save_dir=True
)


def make_train_state(args):
    return {'epoch_index':0,
            'train_loss':[],
            'train_acc':[],
            'val_loss':[],
            'val_acc':[],
            'test_loss':-1,
            'test_acc':-1}
train_state = make_train_state(args)

if not torch.cuda.is_available():
    args.cuda = False
args.device = torch.device("cuda" if args.cuda else "cpu")

# create dataset and vectorizer
dataset = SurnameDataset.load_dataset_and_make_vectorizer(args.surname_csv)
vectorizer = dataset.get_vectorizer()



classifier = SurnameClassifier(embedding_size=args.char_embedding_size, 
                               num_embeddings=len(vectorizer.char_vocab),
                               num_classes=len(vectorizer.nationality_vocab),
                               rnn_hidden_size=args.rnn_hidden_size,
                               padding_idx=vectorizer.char_vocab.mask_index)


classifier = classifier.to(args.device)

dataset.class_weights = dataset.class_weights.to(args.device)
    
loss_fuc = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                           mode='min', factor=0.5,
                                           patience=1)


                                           
train_state = make_train_state(args)

# epoch_bar = tqdm_notebook(desc='training routine', 
                          # total=args.num_epochs,
                          # position=0)

# dataset.set_split('train')
# train_bar = tqdm_notebook(desc='split=train',
                          # total=dataset.get_num_batches(args.batch_size), 
                          # position=1, 
                          # leave=True)
# dataset.set_split('val')
# val_bar = tqdm_notebook(desc='split=val',
                        # total=dataset.get_num_batches(args.batch_size), 
                        # position=1, 
                        # leave=True)
                        
for epoch_index in range(args.num_epochs):
    train_state['epoch_index'] = epoch_index
    
    dataset.set_split('train')
    batch_generator = generate_batches(dataset , batch_size = args.batch_size , device = args.device)
    running_loss = 0.0
    running_acc = 0.0
    correct = 0
    classifier.train()
    for batch_index,batch_dict in enumerate(batch_generator):
        optimizer.zero_grad()
        y_pred = classifier(x_in = batch_dict['x_data'], 
                            x_lengths=batch_dict['x_length'])
        # print(y_pred)
        # print(batch_dict['y_target'].float())
        loss = loss_fuc(y_pred , batch_dict['y_target'])
        loss_batch = loss.item()
        running_loss += (loss_batch-running_loss)/(batch_index + 1)
        loss.backward()
        optimizer.step()
        acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
        running_acc += (acc_t - running_acc) / (batch_index + 1)
        # train_bar.set_postfix(loss=running_loss, acc=running_acc, 
                                  # epoch=epoch_index)
        # train_bar.update()
    train_state['train_loss'].append(running_loss)
    train_state['train_acc'].append(running_acc)
    print("Epoch:" + str(epoch_index) + "  " + "running_loss:" + str(round(running_loss , 4)) + "         " + "running_acc:%" + str(round(running_acc , 2)))
    
    
    
    
    dataset.set_split('val')
    batch_generator = generate_batches(dataset , batch_size = args.batch_size , device = args.device)
    running_loss = 0.0
    running_acc = 0.0
    classifier.eval()
    for batch_index,batch_dict in enumerate(batch_generator):
        # print("batch num: " + str(batch_index))
        y_pred = classifier(x_in = batch_dict['x_data'], 
                                x_lengths=batch_dict['x_length'])
        loss = loss_fuc(y_pred , batch_dict['y_target'])
        loss_batch = loss.item()
        running_loss += (loss_batch-running_loss)/(batch_index + 1)
        acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
        running_acc += (acc_t - running_acc) / (batch_index + 1)
         # train_bar.set_postfix(loss=running_loss, acc=running_acc, 
                                  # # epoch=epoch_index)
        # train_bar.update()        
    train_state['val_loss'].append(running_loss)
    train_state['val_acc'].append(running_acc)        
    print("Epoch:" + str(epoch_index) + "  " + "eval_running_loss:" + str(round(running_loss , 4)) + "    " + "eval_running_acc:%" + str(round(running_acc , 2)))
    print("saving checkpoint")
    torch.save({
            'epoch': epoch_index,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, PATH)   


