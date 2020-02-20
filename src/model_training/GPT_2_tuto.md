#Fonctionnement GPT_2 avec Huggingface

Je détaille ici différents modules de huggingface en me basant sur leur 
example run_language_modelling.py
https://github.com/huggingface/transformers/blob/master/examples/run_language_modeling.py

Ce script est utilisé pour fine-tuner un transformer (BERT, GPT_2, CTRL, ...).

## Data à fournir

### Dans le cas du script 

Ce script attends en entrée un seul et unique fichier texte. 
Ils ont prévu une classe [TextDataset](https://github.com/huggingface/transformers/blob/master/examples/run_language_modeling.py#L86) 
qui fonctionne de la manière suivante
1. Split le texte en bloc de taille block_size (= nb de tokens que le transformer peut prendre en entrée) 
    Il balance à la poubelle le dernier bloc (ils ne s'embêtent pas à padder au moins)
2. Tokenize chaque bloc avec `tokenizer.tokenize`
3. Récupérer les ID de chaque token avec `tokenizer.convert_tokens_to_id`
(comme d'hab en NLP, ie Keras, on donne en entrée du transformer les id) 
4. La liste des blocs est gardé en mémoire sous la forme d'un torch dataset
5. Ce dataset sera ensuite chargé sous la forme d'un dataloader à l'intérieur de la fonction train du script

### Pour notre utilisation 

#### Padder 
Dans notre cas, c'est surement une mauvaise idée d'utiliser un fichier texte en input
car on veut avoir en input de GPT-2
- $P1$ $P3$ meta_data $P2$ 
et c'est tout, on ne veut surtout pas avoir une suite de texte qui va polluer cette première partie.
Dans notre cas, il faudra donc padder.

**Si on veut ré-utiliser le script sans avoir un fournir un fichier txt en entrée,
il faut donc simplement générer un dataset (donc chaque instance est de size block_size * gpt_2_vocab_size) et le balancer à train** 

#### Token_special 
On peut facilement indiquer à GPT_2 que l'on veut utiliser des token_speciales

`self.tokenizer.add_special_tokens({'begin_P1': '[P1]'})
`

A noter que durant le decoding, on peut indiquer à l'interface décodage de huggingface de ne pas sélectionner ces tokens spéciales.
Un truc vachement intéressant. On peut par exemple mettre à la fin de chaque P2 un token spécial <EOS> et lors du decoding
dire à l'interface d'huggingface d'arrêter de décoder lorsque le modèle output ce token spécial. 
## Focus sur GPT-2 huggingface 

Dans le cas de GPT-2, le modèle que l'on va utiliser est [GPT2LMHeadModel](https://github.com/huggingface/transformers/blob/d490b5d5003654f104af3abd0556e598335b5650/src/transformers/modeling_gpt2.py#L511).
Ils ont aussi un modèle GPT2Model avec en output les raws hiddenlayer. 
Pour GPT2LMHeadModel, il y a un layer supplémentaire qui projette les dernières représentations latentes 
dans l'espace du vocabulaire. À noter que GPT2LMHeadModel output des logits pour chaque token possible (et non pas des proba).

## Entrée / Sortie de GPT-2

En fait, en sortie de GPT2LMHeadModel (mais plus généralement pour n'importe quel
transformer model dans huggingface), l'output est un gros tupple 
qui contient à la fois les logits pour chaque token/batch (ce que j'ai parlé au dessus), mais aussi des 
valeurs sur le contenue des attention_head, etc. 

Autre point d'attention : deux possibilités pour appeler GPT2LMHeadModel sur un input.
Soit simplement `GPT2LMHeadModel(input)`, mais aussi `GPT2LMHeadModel(input, labels=input)` 

### Cas du training 
On utilise la possibilité cité ci-dessus pour train. En faisant cela, dans l'output, en plus de tout ce que l'on 
a vu au dessus, on peut aussi récupérer la loss. 

```
model.train()
outputs = else model(inputs, labels=labels)
loss = outputs[0]  
```
dans ce cas la loss se trouve dans le premier élément du tupple. 
Il reste plus qu'à faire `loss.backward()`
Globalement il utilise pas mal de techniques poussées au sein de la fonction train (warm_start, learning_rate_scheduler) et je pense
qu'on a pas intérêt à refaire ça à la main. 

à noter également, que inputs est de shape batch_size * sequence_lenght. Et non pas
batch_size * sequence_length * vocab_size car on envoie les [TokenId](https://huggingface.co/transformers/glossary.html#input-ids)
en input

### Cas du decoding 

#### A la main 
Dans le cas où on veut faire ça à la mano, c'est simple. 
Supposons on a un texte de longueur block_size. On l'a tokenizé avec le 
tokenizer fourni par huggingface.
```
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

# get the predicted next sub-word (in our case, the word 'man')
predicted_index = torch.argmax(predictions[0, -1, :]).item()
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
```
Là comme on a pas fait `model(inputs, labels=labels)` mais `model(input)` le premier élément
du tupple de sortie de model n'est plus loss mais le tensor de sortie de shape
`batch_size * sequence_lenght * vocab_size` Dans notre cas, on a un batch
de size 1, on veut chopper la distribution de proba pour le dernier token d'où `
predictions[0, -1, :]`

#### En utilisant des fonctions d'huggingface

Bien sur huggingface propose des fonctions de décodage pré-codé et qu'on peut vraiment
paramétré en détail. La fonction s'appelle [generate](https://github.com/huggingface/transformers/blob/d490b5d5003654f104af3abd0556e598335b5650/src/transformers/modeling_utils.py#L585)
La [doc](https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.generate) est bien foutue 

Par contre, dans le cas où on veut utiliser cette fonction il faudra penser à padder à gauche avant d'envoyer le texte.
Pourquoi ? Car generate fait une beam search à partir du dernier token de la phrase. 
 