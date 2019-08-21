# hate-speech-and-offensive-language


## ToDo

- Nuove modalità per estrarre le feature da un testo.


- Provare metodi diversi di sentimental analysis.
  

- Ottimizzare stemmer


- Nel progetto usano GridSearch e Kfold (5) per ottimizzare gli hyperparameters di LogisticRegression.
Si può fare la stessa cosa su più classificatori, provare tecniche di ensemblement. Ad esempio Bagging per combinare più classificatori.


- Valutare utilità del feature extraction tramite l'analisi dele Part Of Speech


- This histogram shows the imbalanced nature of the task - most tweets containing "hate" words as defined by Hatebase were only considered to be offensive by the CF coders. More tweets were considered to be neither hate speech nor offensive language than were considered hate speech.
  -Pensare un modo per bilanciare questa cosa
  
  
  
  
## Dubbi: 

- Dove usa: from sklearn.svm import LinearSVC
- Nel progetto usano solo la class, può essere utile anche analizzare in che rapporto hanno votato le persone? (esempio unanimità vs.         51%-49%)


