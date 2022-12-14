import streamlit as st
import streamlit.components.v1 as components  
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
import sklearn
import pandas as pd
import numpy as np
from pandasql import sqldf
import sweetviz as sv
import base64 
import tpot
import html5lib
import requests
import time
import codecs
import os
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
from sklearn.utils._testing import ignore_warnings
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder, JsCode
from lazypredict.Supervised import LazyClassifier
from lazypredict.Supervised import LazyRegressor
from tpot import TPOTClassifier
from tpot import TPOTRegressor
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
estimators = sklearn.utils.all_estimators(type_filter=None)

# Intestazione 
st.set_page_config(page_title="Suite Analisi Dati", page_icon="🔍", layout='wide', initial_sidebar_state='auto')

st.markdown("<center><h1> Italian Intelligence Analytic Suite <small><br> Powered by INTELLIGENZAARTIFICIALEITALIA.NET </small></h1>", unsafe_allow_html=True)
st.write('<p style="text-align: center;font-size:15px;" > <bold>Tutti i tool di Analisi, Pulizia e Visualizzazione Dati in unico Posto <bold>  </bold><p><br>', unsafe_allow_html=True)
hide_st_style = """
			<style>
			#MainMenu {visibility: hidden;}
			footer {visibility: hidden;}
			header {visibility: hidden;}
			</style>
			"""
st.markdown(hide_st_style, unsafe_allow_html=True)

#funzioni utili
def get_binary_file_downloader_html(bin_file, file_label='File'):
	with open(bin_file, 'rb') as f:
		data = f.read()
	bin_str = base64.b64encode(data).decode()
	href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
	return href

def app(dataset):
	# Use the analysis function from sweetviz module to create a 'DataframeReport' object.
	analysis = sv.analyze([dataset,'AnalisiEDA2_powered_by_IAI'], feat_cfg=sv.FeatureConfig(force_text=[]), target_feat=None)
	analysis.show_html(filepath='AnalisiEDA2_powered_by_IAI.html', open_browser=False, layout='vertical', scale=1.0)
	HtmlFile = open("AnalisiEDA2_powered_by_IAI.html", 'r', encoding='utf-8')
	source_code = HtmlFile.read() 
	components.html(source_code,height=1200, scrolling=True)
	st.markdown(get_binary_file_downloader_html('AnalisiEDA2_powered_by_IAI.html', 'Report'), unsafe_allow_html=True)
	st.success("Secondo Report Generato Con Successo, per scaricarlo clicca il Link quì sopra.")
	st.warning("Mi raccomando segnati quanto ti occorre per passare al prossimo step")
	st.info("Ti consigliamo vivamente di scaricare il report per poterlo consultare in un secondo momento")

#inizio body
uploaded_file = st.file_uploader("Perfavore inserisci quì il file di tipo csv, usando come separatore la virgola!", type=["csv"])

if uploaded_file is not None:
    Menu = option_menu("La DataScience per tutti 🐍🔥", ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6"],
                    icons=['clipboard-data', 'globe', 'file-pdf', 'file-earmark-spreadsheet'],
                    menu_icon="app-indicator", default_index=0,orientation='horizontal',
                    styles={
                    "container": {"padding": "5!important", "background-color": "#fafafa", "width": "100%"},
                    "icon": {"color": "blak", "font-size": "15px"}, 
                    "nav-link": {"color": "blak","font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                    "nav-link-selected": {"color": "blak","background-color": "#02ab21"},
                    }
        )
    
    #render del dataset
    dataset = pd.read_csv(uploaded_file)
    colonne = list(dataset.columns)
    options = st.multiselect("Seleziona le colonne che vuoi usare..",colonne,colonne)
    dataset = dataset[options]
    gb = GridOptionsBuilder.from_dataframe(dataset)

    #customize gridOptions
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
    gb.configure_grid_options(domLayout='normal')
    gridOptions = gb.build()
    
    try:
        with st.expander("VISUALIZZA e MODIFICA il DATASET"):
            grid_response = AgGrid(
            dataset, 
            gridOptions=gridOptions,
            enable_enterprise_modules=True,
            update_mode="MODEL_CHANGED",
            )
    
        with st.expander("VISUALIZZA delle STATISICHE di BASE"):
            st.write(dataset.describe())
    except Exception as e:
        print(e)
        st.error("Errore nel caricamento del dataset, perfavore tenta con un altro set di dati o contatta l'assistenza")
    
    if Menu == "Step 1":
        st.title("Step 1 : Analisi Dati 📊")
        with st.expander("A cosa serve questa sezione ?"):
            st.markdown("In questa sezione puoi fare <b>una prima analisi dei dati</b>, per capire se : <br>\
                - Ci sono valori anomali<br> \
                - Ci sono colonne che non servono<br>\
                - Ci sono valori nulli o mancanti<br>\
                - Ci sono valori duplicati<br>\
                per poi poter decidere se fare una pulizia dei dati o no.", unsafe_allow_html=True)
        
        if(st.button("Genera prima Analisi 📊 ")):
            try :
                with st.spinner("Analisi in corso..."):
                    pr = ProfileReport(dataset, explorative=True, orange_mode=False)
                    st_profile_report(pr)
                    pr.to_file("AnalisiEDA_powered_by_IAI.html")
                    st.markdown(get_binary_file_downloader_html('AnalisiEDA_powered_by_IAI.html', 'Report'), unsafe_allow_html=True)
                    st.success("Primo Report Generato Con Successo, per scaricarlo clicca il Link quì sopra.")
                    st.warning("Mi raccomando segnati quanto ti occorre per passare al prossimo step")
                    st.info("Ti consigliamo vivamente di scaricare il report per poterlo consultare in un secondo momento")
                #app(dataset)
                
                st.balloons()
            except Exception as e:
                print(e)
                st.error('Mannaggia, ci dispiace qualcosa non è andato come doveva, riprova')
                
    if Menu == "Step 2":
        st.title("Step 2 : Pulizia Dati 🧹")
        with st.expander("A cosa serve questa sezione ?"):
            st.markdown("In questa sezione puoi fare <b>una veloce pulizia dei dati,</b> ad esempio potrai  : <br> \
                - Rimuovere valori nulli e corrotti <br> \
                - Rimuovere valori duplicati <br>\
                - Sostituire valori nulli con valori medi o mediati <br>\
                per poi poter passare al prossimo step.", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        if col1.button("Rimuovi valori nulli") :
            try:
                dataset = dataset.dropna()
                st.success("Valori nulli rimossi con successo")
                dataset.to_csv('DatasetConValoriNulliRimossi.csv', sep=',', index=False)
                st.markdown(get_binary_file_downloader_html('DatasetConValoriNulliRimossi.csv', 'Scarica i tuoi Dati puliti'), unsafe_allow_html=True)
                st.success("Dati puliti con successo, per scaricarli clicca il Link quì sopra.")
                st.balloons()
            except Exception as e:
                print(e)
                st.error("Mannaggia, ci dispiace qualcosa non è andato come doveva, riprova")
            
        if col2.button("Rimuovi valori duplicati") :
            try:
                dataset = dataset.drop_duplicates()
                st.success("Valori duplicati rimossi con successo")
                dataset.to_csv('DatasetConValoriDuplicatiRimossi.csv', sep=',', index=False)
                st.markdown(get_binary_file_downloader_html('DatasetConValoriDuplicatiRimossi.csv', 'Scarica i tuoi Dati puliti'), unsafe_allow_html=True)
                st.success("Dati puliti con successo, per scaricarli clicca il Link quì sopra.")
                st.balloons()
            except Exception as e:
                print(e)
                st.error("Mannaggia, ci dispiace qualcosa non è andato come doveva, riprova")
            
        if col3.button("Sostituisci valori nulli con valori medi") :
            try:
                dataset = dataset.fillna(dataset.mean())
                st.success("Valori nulli sostituiti con successo")
                dataset.to_csv('DatasetConValoriNulliSostituiti.csv', sep=',', index=False)
                st.markdown(get_binary_file_downloader_html('DatasetConValoriNulliSostituiti.csv', 'Scarica i tuoi Dati puliti'), unsafe_allow_html=True)
                st.success("Dati puliti con successo, per scaricarli clicca il Link quì sopra.")
                st.balloons()
            except Exception as e:
                print(e)
                st.error("Mannaggia, ci dispiace qualcosa non è andato come doveva, riprova facendo attenzioni che i dati nulli non siano testuali o categorici")
            
        if col4.button("Sostituisci valori nulli con valori mediati") :
            try:
                dataset = dataset.fillna(dataset.median())
                st.success("Valori nulli sostituiti con successo")
                dataset.to_csv('DatasetConValoriNulliSostituiti.csv', sep=',', index=False)
                st.markdown(get_binary_file_downloader_html('DatasetConValoriNulliSostituiti.csv', 'Scarica i tuoi Dati puliti'), unsafe_allow_html=True)
                st.success("Dati puliti con successo, per scaricarli clicca il Link quì sopra.")
                st.balloons()
            except Exception as e:
                print(e)
                st.error("Mannaggia, ci dispiace qualcosa non è andato come doveva, riprova facendo attenzioni che i dati nulli non siano testuali o categorici")

    if Menu == "Step 3":
        st.title("Step 3 : PreProcessing 🧪")
        with st.expander("A cosa serve questa sezione ?"):
            st.markdown("In questa sezione puoi <b>eseguire operazioni di Preprocessing sui tuoi dati</b>, ad esempio potrai  : <br>\
                - Fare una codifica OneHotEncoding<br>\
                - Fare una codifica LabelEncoding<br>\
                - Fare una codifica OrdinalEncoding<br>\
                - Fare una codifica BinaryEncoding<br>\
                - Fare una codifica TargetEncoding<br>\
                - Fare una codifica MeanEncoding<br>\
                - Fare una codifica CountEncoding<br>\
                - Fare una codifica FrequencyEncoding<br>\
                - Standardizzare i dati<br>\
                - Normalizzare i dati<br>\
                - Fare una riduzione della dimensionalità<br>\
                - Fare una riduzione della dimensionalità con PCA<br>\
                per poi poter passare a una seconda analisi dove scegliarai le feature più correlate con la variabile target.")
        
        task = [ "Codifica OneHotEncoding", "Codifica LabelEncoding", "Codifica OrdinalEncoding", "Codifica BinaryEncoding", "Codifica TargetEncoding", "Codifica MeanEncoding", "Codifica CountEncoding", "Codifica FrequencyEncoding"]
        colonna_da_codificare = st.selectbox("Seleziona la colonna da codificare", dataset.columns)
        task = st.selectbox("Seleziona la codifica da eseguire", task)
        if st.button("Codifica"):
            try:
                from sklearn import preprocessing
                if task == "Codifica OneHotEncoding":
                    dataset = pd.get_dummies(dataset, columns=[colonna_da_codificare])
                    st.success("Codifica eseguita con successo")
                    dataset.to_csv('DatasetCodificato.csv', sep=',', index=False)
                    st.markdown(get_binary_file_downloader_html('DatasetCodificato.csv', 'Scarica i tuoi Dati Codificati'), unsafe_allow_html=True)
                    st.success("Dati Codificati con successo, per scaricarli clicca il Link quì sopra.")
                    st.balloons()
                if task == "Codifica LabelEncoding":
                    le = preprocessing.LabelEncoder()
                    dataset[colonna_da_codificare] = le.fit_transform(dataset[colonna_da_codificare])
                    st.success("Codifica eseguita con successo")
                    dataset.to_csv('DatasetCodificato.csv', sep=',', index=False)
                    st.markdown(get_binary_file_downloader_html('DatasetCodificato.csv', 'Scarica i tuoi Dati Codificati'), unsafe_allow_html=True)
                    st.success("Dati Codificati con successo, per scaricarli clicca il Link quì sopra.")
                    st.balloons()
                if task == "Codifica OrdinalEncoding":
                    oe = preprocessing.OrdinalEncoder()
                    dataset[colonna_da_codificare] = oe.fit_transform(dataset[colonna_da_codificare])
                    st.success("Codifica eseguita con successo")
                    dataset.to_csv('DatasetCodificato.csv', sep=',', index=False)
                    st.markdown(get_binary_file_downloader_html('DatasetCodificato.csv', 'Scarica i tuoi Dati Codificati'), unsafe_allow_html=True)
                    st.success("Dati Codificati con successo, per scaricarli clicca il Link quì sopra.")
                    st.balloons()
                if task == "Codifica BinaryEncoding":
                    be = preprocessing.BinaryEncoder()
                    dataset[colonna_da_codificare] = be.fit_transform(dataset[colonna_da_codificare])
                    st.success("Codifica eseguita con successo")
                    dataset.to_csv('DatasetCodificato.csv', sep=',', index=False)
                    st.markdown(get_binary_file_downloader_html('DatasetCodificato.csv', 'Scarica i tuoi Dati Codificati'), unsafe_allow_html=True)
                    st.success("Dati Codificati con successo, per scaricarli clicca il Link quì sopra.")
                    st.balloons()
                if task == "Codifica TargetEncoding":
                    te = preprocessing.TargetEncoder()
                    dataset[colonna_da_codificare] = te.fit_transform(dataset[colonna_da_codificare])
                    st.success("Codifica eseguita con successo")
                    dataset.to_csv('DatasetCodificato.csv', sep=',', index=False)
                    st.markdown(get_binary_file_downloader_html('DatasetCodificato.csv', 'Scarica i tuoi Dati Codificati'), unsafe_allow_html=True)
                    st.success("Dati Codificati con successo, per scaricarli clicca il Link quì sopra.")
                    st.balloons()
                if task == "Codifica MeanEncoding":
                    me = preprocessing.MeanEncoder()
                    dataset[colonna_da_codificare] = me.fit_transform(dataset[colonna_da_codificare])
                    st.success("Codifica eseguita con successo")
                    dataset.to_csv('DatasetCodificato.csv', sep=',', index=False)
                    st.markdown(get_binary_file_downloader_html('DatasetCodificato.csv', 'Scarica i tuoi Dati Codificati'), unsafe_allow_html=True)
                    st.success("Dati Codificati con successo, per scaricarli clicca il Link quì sopra.")
                    st.balloons()
                if task == "Codifica CountEncoding":
                    ce = preprocessing.CountEncoder()
                    dataset[colonna_da_codificare] = ce.fit_transform(dataset[colonna_da_codificare])
                    st.success("Codifica eseguita con successo")
                    dataset.to_csv('DatasetCodificato.csv', sep=',', index=False)
                    st.markdown(get_binary_file_downloader_html('DatasetCodificato.csv', 'Scarica i tuoi Dati Codificati'), unsafe_allow_html=True)
                    st.success("Dati Codificati con successo, per scaricarli clicca il Link quì sopra.")
                    st.balloons()
                if task == "Codifica FrequencyEncoding":
                    fe = preprocessing.FrequencyEncoder()
                    dataset[colonna_da_codificare] = fe.fit_transform(dataset[colonna_da_codificare])
                    st.success("Codifica eseguita con successo")
                    dataset.to_csv('DatasetCodificato.csv', sep=',', index=False)
                    st.markdown(get_binary_file_downloader_html('DatasetCodificato.csv', 'Scarica i tuoi Dati Codificati'), unsafe_allow_html=True)
                    st.success("Dati Codificati con successo, per scaricarli clicca il Link quì sopra.")
                    st.balloons()
            except Exception as e:
                st.error("Errore durante la codifica, controlla di aver selezionato la colonna e la codifica da eseguire")
                st.error(e)
                
            ##DA AGGIUNGERE STANDARDIZZAZIONE E MINIMIZZAZIONE
            q = st.text_input("Scrivi qui dentro la tua Query", value="SELECT * FROM dataset")
            if st.button("Applicami questa Query SQL sui miei dati"):
                try :
                    df = sqldf(q)
                    df = pd.DataFrame(df)
                    st.write(df)
                    df.to_csv("RisultatiQuery_powered_by_IAI.csv")
                    st.markdown(get_binary_file_downloader_html('RisultatiQuery_powered_by_IAI.csv', 'Riusltato Query Sql sui tuoi dati'), unsafe_allow_html=True)
                    st.balloons()
                except Exception as e:
                    print(e)
                    st.error('Mannaggia, ci dispiace qualcosa non è andato come doveva, riprova')
        
    if Menu == "Step 4" :
        st.title("Step 4 : Seconda Analisi dei Dati 📊")
        with st.expander("A cosa serve questa sezione ?"):
            st.markdown("In questa sezione puoi fare la <b>seconda analisi dei tuoi dati una volta pre elaborati</b>, questa sezione ti servirà per : <br>\
                - Capire se i tuoi dati sono bilanciati o no<br> \
                - Capire se i tuoi dati sono normalmente distribuiti o no<br> \
                - Capire se i tuoi dati sono correlati o no<br> \
                - Capire se i tuoi dati sono outliers o no<br> \
                - Capire se i tuoi dati sono lineari o no<br> \
                - Capire se i tuoi dati sono gaussiani o no <br>\
                - Capire quali saranno le variabili più importanti per il modello<br> \
                - Capire quali variabili saranno inutili per il modello<br> \
                per poi poter scoprire quale sarà il miglior algoritmo per il tuo dataset", unsafe_allow_html=True)
        
        if st.button("Procedi con la nuova analisi 📊"):
            try:
                with st.spinner("Analisi in corso..."):
                    app(dataset)
                    
            except Exception as e:
                st.error("Errore durante l'analisi, controlla di aver caricato il dataset")
                print(e)
            
    if Menu == "Step 5":
        st.title("Step 5 : Scopri il Miglior Algoritmo 🤖")
        with st.expander("A cosa serve questa sezione ?"):
            st.markdown("In questa sezione puoi scoprire quale sarà il miglior algoritmo per il tuo dataset, questa sezione ti servirà per : <br>\
                - Capire quale sarà il miglior algoritmo per il tuo dataset <br>\
                per poi poterlo implementare o farlo implementare automaticamente nel prossimo step <br>", unsafe_allow_html=True)
        
        datasetMalgo = dataset
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        datasetMalgo = datasetMalgo.select_dtypes(include=numerics)
        datasetMalgo = datasetMalgo.dropna()
        colonne = datasetMalgo.columns
        target = st.selectbox('Scegli la variabile Target', colonne )
        st.write("target impostato su " + str(target))
        datasetMalgo = datasetMalgo.drop(target,axis=1)
        colonne = datasetMalgo.columns
        descrittori =  st.multiselect('Scegli la variabili Indipendenti', colonne )
        st.write("Variabili Indipendenti impostate su  " + str(descrittori))
        
        problemi = ["CLASSIFICAZIONE", "REGRESSIONE" ]
        tipo_di_problema = st.selectbox('Che tipo di Algortimo devi utilizzare sui tuoi dati ?', problemi)
        percentuale_dati_test = st.slider('Seleziona la percentuale di dati per il Test', 0.1, 0.9, 0.25)
    
        X = dataset[descrittori]
        y = dataset[target]
    
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=percentuale_dati_test)
    
        if(st.button("Svelami il Miglior Algoritmo per i miei dati Gratuitamente")):
            #try :
                
            if(tipo_di_problema == "CLASSIFICAZIONE"):

                with st.spinner("Dacci qualche minuto, stiamo provando tutti gli algoritmi di Classificazione sui tuoi dati. Se il file è molto grande > 100.000 righe, potrebbe volerci qualche minuto in più."):
                    try:
                        clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
                        models,predictions = clf.fit(X_train, X_test, y_train, y_test)
                        st.table(models)
                        models = pd.DataFrame(models)
                        models.to_csv("MigliorAlgoritmo_powered_by_IAI.csv")
                        st.markdown(get_binary_file_downloader_html('MigliorAlgoritmo_powered_by_IAI.csv', 'Rapporto Modelli Predittivi'), unsafe_allow_html=True)
                        st.balloons()
                    except Exception as e:
                        print(e)
                        st.error('Mannaggia, ci dispiace qualcosa non è andato come doveva. Prova a ridimensionare o campionare il tuo dataset, oppure a cambiare il tipo di problema (Classificazione o Regressione).')

            if(tipo_di_problema == "REGRESSIONE"):

                with st.spinner("Dacci un attimo, stiamo provando tutti gli algoritmi di Regressione sui tuoi dati"):
                    try:
                        reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
                        models, predictions = reg.fit(X_train, X_test, y_train, y_test)
                        st.table(models)
                        models = pd.DataFrame(models)
                        models.to_csv("MigliorAlgoritmo_powered_by_IAI.csv")
                        st.markdown(get_binary_file_downloader_html('MigliorAlgoritmo_powered_by_IAI.csv', 'Rapporto Modelli Predittivi'), unsafe_allow_html=True)		
                        st.balloons()
                    except Exception as e:
                        print(e)
                        st.error('Mannaggia, ci dispiace qualcosa non è andato come doveva. Prova a ridimensionare o campionare il tuo dataset, oppure a cambiare il tipo di problema (Classificazione o Regressione).')

    if Menu == "Step 6":
        st.title("Step 6 : Implementazione Algoritmo 🤖")
        with st.expander("A cosa serve questa sezione ?"):
            st.markdown("In questa sezione puoi implementare il miglior algoritmo per il tuo dataset, questa sezione ti servirà per :<br> \
                - Implementare il miglior algoritmo per il tuo dataset<br> \
                per poi poterlo utilizzare per fare predizioni", unsafe_allow_html=True)	
        
        datasetPalgo = dataset
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        datasetPalgo = datasetPalgo.select_dtypes(include=numerics)
        datasetPalgo = datasetPalgo.dropna()
        colonne = datasetPalgo.columns
        target = st.selectbox('Scegli la variabile Target', colonne )
        st.write("target impostato su " + str(target))
        datasetPalgo = datasetPalgo.drop(target,axis=1)
        colonne = datasetPalgo.columns
        descrittori =  st.multiselect('Scegli la variabili Indipendenti', colonne )
        st.write("Variabili Indipendenti impostate su  " + str(descrittori))
        
        problemi = ["CLASSIFICAZIONE", "REGRESSIONE" ]
        tipo = st.selectbox('Che tipo di Algortimo devi utilizzare sui tuoi dati ?', problemi)
        percentuale_dati_test = st.slider('Seleziona la percentuale di dati per il Test', 0.1, 0.9, 0.25)
    
        gen = st.slider('GENERAZIONI : Numero di iterazioni del processo di ottimizzazione della pipeline di esecuzione. Deve essere un numero positivo o Nessuno.', 1, 10, 5)
        pop = st.slider('POPOLAZIONE : Numero di dati da mantenere nella popolazione di programmazione genetica in ogni generazione.', 1, 150, 20)
        
        scor = ['accuracy', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss', 'precision']
        sel_scor = st.selectbox('Che tipo di metrica vuoi che sia utilizzato ? Se non conosci questi metodi inserisci "accuracy"', scor)
    
        X = dataset[descrittori]
        y = dataset[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=percentuale_dati_test)
    
        if(st.button("Creami la miglior pipeline in Python Gratuitamente")):
            #try :
                
            if tipo=="CLASSIFICAZIONE":
                
                with st.spinner("Dacci qualche minuto, stiamo scrivendo il Codice in Python che implementa il miglior algoritmo sui tuoi dati e ottimizzandolo con gli iperparametri. Maggiore è il numero di Generazioni e Popolazione maggiore sarà il tempo di ATTESA... "):
                    try:
                        pipeline_optimizer = TPOTClassifier()
                        pipeline_optimizer = TPOTClassifier(generations=gen, population_size=pop, scoring=sel_scor, cv=5,
                                random_state=42, verbosity=2)
                        pipeline_optimizer.fit(X_train, y_train)
                        pipeline_optimizer.export('PipelinePython_powered_by_IAI.py')
                        filepipeline = open("PipelinePython_powered_by_IAI.py", 'r', encoding='utf-8')
                        source_code = filepipeline.read() 
                        st.subheader("Miglior PipeLine Rilevata Sui tuoi Dati ")
                        my_text = st.text_area(label="Hai visto, Scriviamo anche il codice al posto tuo...", value=source_code, height=500)
                        st.markdown(get_binary_file_downloader_html('PipelinePython_powered_by_IAI.py', 'Scarica il file python pronto per essere eseguito'), unsafe_allow_html=True)
                        st.balloons()
                    except Exception as e:
                        print(e)
                        st.error('Mannaggia, ci dispiace qualcosa non è andato come doveva. Riprova cambiando il tipo di metrica, o il numero di Generazioni e Popolazione, o il tipo di problema')

            if tipo=="REGRESSIONE":

                with st.spinner(" Dacci qualche minuto, stiamo scrivendo il Codice in Python che implementa il miglior algoritmo sui tuoi dati e ottimizzandolo con gli iperparametri. Maggiore è il numero di Generazioni e Popolazione maggiore sarà il tempo di ATTESA..."):
                    try:
                        pipeline_optimizer = TPOTRegressor()
                        pipeline_optimizer = TPOTRegressor(generations=gen, population_size=pop, scoring=sel_scor, cv=5,
                                random_state=42, verbosity=2)
                        pipeline_optimizer.fit(X_train, y_train)
                        #st.write(f"Accuratezza PIPELINE : {pipeline_optimizer.score(X_test, y_test)*100} %")
                        pipeline_optimizer.export('PipelinePython_powered_by_IAI.py')
                        filepipeline = open("PipelinePython_powered_by_IAI.py", 'r', encoding='utf-8')
                        source_code = filepipeline.read() 
                        st.subheader("Miglior PipeLine Rilevata Sui tuoi Dati ")
                        my_text = st.text_area(label="Hai visto, Scriviamo anche il codice al posto tuo...", value=source_code, height=500)
                        st.markdown(get_binary_file_downloader_html('PipelinePython_powered_by_IAI.py', 'Scarica il file python pronto per essere eseguito'), unsafe_allow_html=True)
                        st.balloons()
                    except Exception as e:
                        print(e)
                        st.error('Mannaggia, ci dispiace qualcosa non è andato come doveva. Riprova cambiando il tipo di metrica, o il numero di Generazioni e Popolazione, o il tipo di problema')
            