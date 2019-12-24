
#
### Progetto Data Mining – Hate speech and offensive language recognition.

Di Russo Mattia, 725516



#
# 1 Introduzione


Lo scopo del progetto è quello di affrontare un problema di Natural Language Processing, applicando tecniche di sentiment analysis per distinguere in maniera automatica frasi offensive (offensive sentences) da frasi che generano odio (hate sentences). Un problema preliminare che notiamo è quello di dare una definizione formale di hate sentence. In generale possiamo dire che è un tipo di linguaggio che viene indirizzato verso gruppi sociali svantaggiati e che potrebbe promuovere violenza o disordini sociali (razzismo, sessismo, omofobia, ecc. ).

Questa differenza tra i due tipi di linguaggi è demarcata nelle legislature di alcuni stati, dove l&#39;utilizzo di un linguaggio di odio viene punito dalla legge; queste regole si estendono anche i social network che hanno dovuto rispondere a questo problema istituendo policy che vietano l&#39;utilizzo di frasi di odio, e rendendo necessaria l&#39;ideazione di meccanismi di rilevamento automatico come quello che si è cercato di realizzare in questo progetto.

Analizzeremo nei capitoli successivi le tecniche di sentiment analysis utilizzate per ottenere il riconoscimento automatico di questi tre tipi di sentences:

- --Analisi del dataset
- --Preprocessing dei dati
- --Feature extraction
- --Training e test dei modelli

#
# 2 Analisi del Dataset



Il dataset utilizzato in questo progetto contiene un corpus di tweet estratti da Twitter, scegliendo i quelli che contenessero parole selezionate da un elenco di parole di odio, reperibile sul sito hatebase.org.

In seguito, un gruppo di utenti ha assegnato a ogni tweet una tra le seguenti label: Hate, Offensive, Neither. Ogni tweet è stato &quot;votato&quot; da tre o più utenti, e ad ogni tweet è stata assegnata la label definitiva utilizzando una decisione per maggioranza; i tweet per i quali non è stata trovata una maggioranza sono stati scartati dal dataset.

 IMMAGINE

Figura A - Import del dataset e tail() del DataFrame associato

La colonna **Unnamed:0** contiene l&#39;id originale del tweet, che nel dataset importato ha un id minore: questo perché i dati per i quali non è stata trovata una maggioranza di voti sono stati scartati.

La colonna **count** contiene il numero di quante persone hanno votato un determinato tweet, mentre **hate\_speech** , **offensive\_language** e **neither** rappresentano il numero di voti ricevuti dal tweet per ogni classe. L&#39;attributo **class** contiene la label attribuita al tweet per maggioranza.

Infine, la colonna **tweet** contiene il messaggio del tweet stesso.

E&#39; possibile notare in Figura B una larga discrepanza nel numero di tweet assegnati a ogni label: solo il 5% dei tweet sono codificati come &quot;Hate&quot;, il 16% sono &quot;Neither&quot;, mentre i restanti sono &quot;Offensive&quot;.

 IMMAGINE

Figura B - Distribuzione delle classi

Poiché questa distribuzione non omogenea potrebbe ridurre notevolmente il recall, abbiamo scelto di utilizzare tecniche di sampling per rendere il dataset più equilibrato; per fare ciò, iniziamo applicando undersampling (ovvero eliminando dal dataset elementi che fanno parte della classe più popolata) sui tweet con classe offensive\_language: creiamo un nuovo DataFrame che conterrà circa la metà dei tweet catalogati come offensivi e tutti i tweet delle altre classi. Questa riduzione nel numero dei dati dovrebbe portare alla creazione di un modello che ha una precisione generale più bassa, ma che riconosce in maniera più efficace le occorrenze delle due classi in minoranza. Cerchiamo quindi di trovare un equilibrio tra il recall (percentuale di istanze da riconoscere – nel nostro caso di tipo hate – che sono classificate come tali) e la precisione (percentuale di classificazioni esatte). Migliorare il primo a discapito della seconda, è generalmente una tecnica utile per riconoscere istanze della classe in minoranza.

Come anticipato, abbiamo eliminato _circa_ la metà dei tweet offensivi. E&#39; stato realizzato ciò rimuovendo tutti i messaggi di classe 1 (offensive\_language) di indice pari. E&#39; stato preferito questo approccio piuttosto che eliminarli in maniera random per rendere il risultato di questo lavoro deterministico, e non dipendente da quali specifici tweet sono stati esclusi in ogni esecuzione differente.

 IMMAGINE

Figura C - Implementazione undersampling

IMMAGINE

Dal grafico della distribuzione delle classi in df1, notiamo un minore sbilanciamento – seppur consistente – nella ripartizione delle varie label. Abbiamo deciso quindi di utilizzare anche una tecnica di oversampling chiamata SMOTE (Synthetic Minority Oversampling Technique), che non crea duplicati, ma genera degli esemplari che hanno caratteristiche simili agli elementi delle classi in minoranza.

SMOTE realizza ciò scegliendo dei record, e modificandone una feature alla volta con un valore casuale, compatibile con quelli dei neighbors di tale record. Poiché stiamo lavorando con un dataset di tweet, quindi testi, questa operazione può essere svolta solo dopo le operazioni di preprocessing e feature extraction. Inoltre, da alcune ricerche su SMOTE, è risultato che è meglio applicare questo metodo dopo aver separato il dataset in test e training data per evitare overfitting.

IMMAGINE

Figura D - Risultato oversampling

##


# 3 Preprocessing dei Dati



Osserviamo in questo capitolo il preprocessing dei dati: tecniche preliminari che ci consentono di &quot;ripulire&quot; i tweet da tutti gli elementi che creano rumore e che non sono utili ai fini della sentiment analysis. Ne sono un esempio i segni di punteggiatura, le citazioni (ad es. @mayasolovely), ma anche tutte quelle parole che sono molto frequenti nel testo, ma non danno un significato specifico alla singola frase come ad esempio le congiunzioni, i pronomi ecc. . Questo gruppo di parole prende il nome di **stop words** , e saranno rimosse da ogni tweet.

Un&#39;altra tecnica importante è quella di **tokenizzazione** che consiste nel dividere ogni frase in un array di singole parole – o tokens – poiché questa struttura è utile per le fasi successive.

Infine, l&#39;ultima tecnica per quanto riguarda il preprocessing è lo **stemming** che consiste nel rimuovere il suffisso da ogni parola, in modo tale da ottenere parole uguali da parole simili che hanno lo stesso significato.



 IMMAGINE
Figura E - Esempio di tweets

# 3.1  Rimozione punteggiatura, numeri e caratteri speciali

Come anticipato, questi tipi di simboli non danno nessun valore aggiunto per la sentiment analysis, anzi causano rumore all&#39;interno dei testi e quindi vanno rimossi.
Rimuoviamo anche gli url e le citazioni ad altri utenti, che non sono altresì rilevanti.

Per fare ciò è sono state utilizzate delle espressioni regolari;  i pattern specificati all&#39;interno del testo vengono riconosciuti ed eliminati.

 IMMAGINE

Figura F - Funzione di ripulitura testo da url e caratteri speciali

Il risultato di questa operazione è una lista di tweet ripulita da tutti gli elementi inutili ai fini della sentiment analysis.

 IMMAGINE

# 3.2  Tokenizing e rimozione Stop Words

Lo step successivo è quello di suddividere il testo in token. Per fare ciò abbiamo utilizzato la libreria nltk (Natural Language Toolkit), che fornisce molti strumenti per quanto riguarda il processing di frasi in linguaggio naturale. Per tokenizzare i tweet abbiamo utilizzato il modulo word\_tokenize di nltk: tramite la funzione tokenize, ogni tweet viene trasformato in un array di tokens.

IMMAGINE

Figura G - Tokenizzazione dei tweet



In questa fase abbiamo scelto anche di rimuovere le stop words, per risparmiare un ciclo for su tutto il dataset. Le stop words non danno del significato aggiunto  alle frase, al contrario rischiano di creare confusione all&#39;interno del dataset. L&#39;elenco di stop words è stato importato sempre dalla libreria nltk, e abbiamo aggiunto a tale elenco un&#39;altra stop word, ovvero &quot;rt&quot;, che è molto presente all&#39;interno dei tweet e corrisponde alla parola &quot;retweet&quot;, che non è rilevante ai fini della sentiment analysis.



IMMAGINE
Figura H - Stop words

In seguito i due grafici delle parole maggiormente presenti all&#39;interno del corpus di tweet. Il primo prende in considerazione tutte le parole, il secondo esclude le stop words. Possiamo notare dal primo grafico che 9 delle 10 parole maggiormente diffuse, sono stop words.

 IMMAGINE

Figura I - Parole più frequenti includendo ed escludendo le stop words



# 3.3 Stemming

L&#39;ultima fase di questo preprocessing consiste nello stemming, ovvero la pratica di ridurre tutte le parole alla loro radice, rimuovendone quindi il suffisso. Ad esempio, parole come &quot;played&quot;, &quot;playing&quot;, &quot;playin&#39;&quot; vengono ridotte tutte alla loro radice &quot;play&quot;. Anche in questo caso abbiamo utilizzato la libreria nltk attraverso il modulo SnowballStemmer. Esistono altre versioni di stemmer, come ad esempio il PorterStemmer che lavora in maniera molto simile allo Snowball (gli stessi realizzatori di Porter hanno ammesso che lo Snowball può essere considerato come una sua evoluzione), e il LancasterStemmer che abbiamo scelto di non utilizzare perché troppo &quot;aggressivo&quot;.

IMMAGINE

Figura J - Stemming dei token

Abbiamo utilizzato un processo di stemming e non lemmatizzazione, poiché quest&#39;ultimo metodo, sebbene produca sempre parole esistenti nel vocabolario, è  più lento rispetto allo stemmer che non si preoccupa di generare parole di senso compiuto.



#
# 4 Feature extraction



Uno dei passi più importanti è quello di feature extraction dal corpus di tweet, che consiste nel convertire un testo in un insieme di valori numerici che lo rappresentano.

Un approccio per fare ciò è la **term frequency-inverse document frequency (TF–IDF)**, che attribuisce un peso ad ogni parola in base a quanto spesso la parola ricorre all&#39;interno del nostro corpus di tweet. TF-IDF è il prodotto tra la **term frequency** e la **inverse**** document ****frequency** :



- **tf:** è la frequenza di una parola all&#39;interno di un singolo tweet. Si ottiene dividendo il numero di volte che la parola i occorre nel documento j, diviso la lunghezza del documento j.

IMMAGINE

- **idf:** indica la quantità di informazione fornita da una parola, in base a quanto è comune tra tutti i tweet. E&#39; il rapporto tra il numero di documenti (tweet) e il numero di volte che la parola i appare tra tutti i documenti.

IMMAGINE

# 4.1  TfidfVectorizer

Abbiamo utilizzato la classe TfidfVectorizer del modulo sklearn, che combina le funzionalità fornite da

- **CountVectorizer** : prende un array di documenti (tweets) e costruisce un modello bag-of-words, ossia un dizionario che associa ad ogni parola il numero di volte che questa parola occorre in tutti i documenti.
- **TfidfTransformer** : prende le frequenze generate da CountVectorizer come input, e le trasforma in tf-idf.

 

Figura K - Matrice tf-idf

Instanziamo un oggetto **TfidfVectorizer** settando innanzitutto i parametri **tokenizer** e **preprocessor** , ai quali assegnamo rispettivamente le funzioni di preprocessing e tokenizing, leggermente modificate rispetto a quelle viste in precedenza. Queste funzioni operano esattamente allo stesso modo, ma prendono come input un singolo tweet invece che una lista.

Il parametro **ngram\_range** indica appunto il range di n-grams che si vuole utilizzare. Un n-gram è una sotto sequenza di un documento composta da n parole. Nel nostro caso verranno presi in considerazione unigram, bigram e trigram.
Infine abbiamo deciso, tramite il parametro **max\_features** , di limitare il numero di features utilizzate per la creazione della matrice a 10000. Questa scelta è stata influenzata dal fatto che un numero più alto di feature portava spesso a un errore di tipo Memory Error durante l&#39;esecuzione del metodo vectorizer.fit\_transform(). Questo valore può essere incrementato se il programma è eseguito su macchine con una RAM più capiente; la nostra intenzione in ogni caso era quella di escludere le features meno rilevanti specificando il parametro max\_features.

# 4.2  POS tag vectorization

Un altro metodo di feature extraction da un testo è quello del POS tagging (Part Of Speech tagging). Tramite questa tecnica si estrae un array di tags a partire da un documento. Questi tags sono relativi a quale parte del discorso (part of speech) appartiene ogni parola, ad esempio sostantivo, verbo, aggettivo, ecc.

 

Figura L - POS tagging

La funzione POS\_tag prende come input un corpus di documenti (tweets) e restituisce un array contenente i POS tags relativi a ogni documento. Per assegnare a ogni parola il suo corrispettivo tag abbiamo utilizzato la funzione pos\_tag della libreria nltk che prende come input un array di token: è stato necessario creare una versione modificata del tokenizer che evitasse di applicare lo stemming alle parole da taggare, poiché dopo aver eseguito lo stemming è impossibile riconoscere ad esempio un nome da un verbo.

Possiamo utilizzare TfidfVectorizer per estrarre le frequenze dei tag relative ad ogni tweet. Evitiamo quindi di calcolare la inverse document frequency in quanto questa informazione non è utile se applicata sui POS tags, le informazioni di questi ultimi sono utili all&#39;interno del contesto di un singolo tweet.



 
Figura M - Matrice tf-idf dei POS tag



# 4.3  Altre features

E&#39; stata realizzata infine una funzione che estrae altre features deducibili direttamente dal testo, ad esempio la lunghezza di ogni tweet, il numero di caratteri del testo preprocessato e non, il numero di sillabe, e altre informazioni di questo tipo. Abbiamo inoltre rilevato se il tweet è un retweet cercandone all&#39;interno la parola &#39;RT&#39; (all&#39;interno del tweet non preprocessato).

Abbiamo utilizzato inoltre un sentiment analyzer del modulo VaderSentiment per aggiungere questo tipo di feature alla nostra matrice.


Figura N - Sentiment Analyzer



Infine abbiamo utilizzato come features anche due parametri che fanno parte dei test di leggibilità Flesch–Kincaid. Ne esistono due tipi di test:

- FLE (Flesch Reading Ease): test che indica quanto è comprensibile un testo. Un alto punteggio a questo test indica una elevata leggibilità, viceversa un basso punteggio corrisponde ad una scarsa leggibilità. Il punteggio è calcolato nel seguente modo:

 

- FK Grade Level: test che usa gli stessi parametri del FLE per calcolare la leggibilità, ma ne è inversamente proporzionale: in questo caso un basso punteggio corrisponde ad un&#39;elevata leggibilità.

 


Mostriamo in Figura O la funzione che estrae queste features da ogni tweet.

 
Figura O - Altre features



# 4.4  Unire tutte le features

Concateniamo queste matrici di features, creando il nostro dataset definitivo. Ricordiamo che questo dataset non contiene i tweet che erano stati rimossi con underfitting.

 

Figura P - Matrice unione di tutte le features estratte



#
# 5 Training e test dei modelli



Per la scelta del modello ideale al nostro scopo sono stati presi in considerazione i seguenti classificatori:

- LogisticRegression

 
- Bernoulli Naïve Bayes

 
- DecisionTree

 

- Support Vector Machine

 

I modelli sono stati addestrati mediante la funzione cross\_validate, che divide il dataset in un numero di fold passato come parametro e successivamente esegue l&#39;addestramento mediante cross validation.

I modelli sono stati valutati in base a due fattori:

- Tempo di fitting del modello

- Accuratezza delle predictions

Per quanto riguarda il tempo di fitting possiamo notare che il DecisionTreeClassifier (DTC) impiega un tempo decisamente lungo probabilmente a causa dell&#39;elevato numero di features (ricordiamo che abbiamo preso in considerazione solo le 10000 features più rilevanti) da analizzare a ogni split dell&#39;albero di decisione.

I modelli probabilistici LogisticRegression (LR) e BernoulliNaiveBayes (BNB) hanno impiegato un tempo decisamente minore, dovuto al minor numero di iterazioni presenti all&#39;interno di questi algoritmi, soprattutto nel caso di BNB.

Infine, SupportVectorClassifier (SVC) non ha terminato la sua esecuzione in tempi ragionevoli, pertanto abbiamo deciso di non utilizzarlo ma comunque di analizzarne il risultato. Questo tipo di classificatore ha una complessità elevata, soprattutto in presenza di un ingente numero di features e dati da analizzare, quindi non è adatto alle nostre esigenze.


Figura Q - Analisi dei tempi di fitting (l&#39;asse y rappresenta i secondi)

Analizziamo adesso i risultati per quanto riguarda l&#39;accuratezza delle predictions. Dalle confusion matrix in Figura R, possiamo notare che BNB e LR presentano risultati simili; BNB è molto più preciso nel classificare i messaggi offensivi e neutri – che non è l&#39;obiettivo di questo progetto – mentre LR riconosce in maniera leggermente più precisa i messaggi di odio, ma con una precisione generale minore.

DTC ha ottenuto i risultati migliori, che sono molto simili a quelli ottenuti dalla ricerca dalla quale questo progetto prende ispirazione, dove però il modello finale utilizzato è Logistic regression. La precisione verso le predizioni sulle classi neither e offensive è molto alta, attestandosi attorno al 90%, mentre quella verso le predizioni di messaggi di odio arriva al 59%.


Figura R - confusion matrix di BernoulliNaiveBayes, LogisticRegression e DecisionTreeClassifier



In conclusione possiamo affermare che il modello realizzato con DecisionTreeClassifier è abbastanza affidabile, ma non essendo stato effettuato per motivi di hardware a disposizione alcun metodo di tuning degli hyperparameters, è legittimo pensare che anche gli altri due classificatori potessero esprimere migliori potenzialità. Questo è avvalorato dal fatto che, come anticipato, gli autori della ricerca sull&#39; hate speech detection hanno ottenuto risultati simili (migliori) utilizzando LogistcRegression come classificatore.

Sarebbe inoltre interessante come sviluppo futuro utilizzare metodi di ensemble per far collaborare più modelli in modo tale da ottenere predictions ancora più precise soprattutto per quanto riguarda il rilevamento di istanze per quanto riguarda la classe Hate, che presenta le maggiori criticità venendo riconosciuta solo il 60% delle volte circa.
