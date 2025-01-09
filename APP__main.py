import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
import streamlit as st 
import numpy as np 
import pandas as pd
import streamlit as st
import google.generativeai as genai
import time
import pandas as pd
import streamlit as st
import altair as alt

# streamlit run C:\\Users\\keno\\OneDrive\\Documents\\Projects\\SPARK-ITB\\APP__main.py

# Create the sidebar
with st.sidebar:
    st.subheader("Select Service of Choice")
    option = st.radio(
        "Select Page:",
        ("Outlier Detection", "Statistics Plotting", "Automated Report"),
        index=0  # Set default to "Cool"
    )

# Display the corresponding message based on the selected option

# Outlier Detection =======================================================================================================================================
if option == "Outlier Detection":
    # Load data
    id = "1R7tyi0-l8Go1xsaKeyAQXNCOu3V-HwT7pJy_VgPPAGE"
    url = f'https://docs.google.com/spreadsheet/ccc?key={id}&output=xlsx'
    df = pd.read_excel(url, sheet_name=0)

    # Create Streamlit app
    st.title("Outlier Detection")

    # Select columns
    columns = df.columns.tolist()
    anomaly_inputs = st.multiselect("Select columns", columns, default=columns)

    # Preprocess IP address columns
    for col in anomaly_inputs[:]:  
        if "IP" in col:
            df[col] = df[col].apply(lambda x: [int(i) for i in x.split('.')])
            new_cols = pd.DataFrame(df[col].tolist(), columns=[f"{col}_{i}" for i in range(1, 5)])
            df = pd.concat([df.drop(col, axis=1), new_cols], axis=1)
            anomaly_inputs.remove(col)
            anomaly_inputs.extend([f"{col}_{i}" for i in range(1, 5)])

    # Create df_final without string columns
    df_final = df.select_dtypes(exclude=['object'])
    anomaly_inputs_final = [col for col in anomaly_inputs if col in df_final.columns]

    # Model parameters
    contamination = st.slider("Contamination", min_value=0.01, max_value=0.5, step=0.01, value=0.1)
    random_state = st.slider("Random State", min_value=0, max_value=42, value=42)

    # Outlier plot function
    def outlier_plot(data, outlier_method_name):
        print(f'Outlier Method: {outlier_method_name}')
        method = f'{outlier_method_name}_anomaly'
        st.write(f"Number of anomalous values {len(data[data['anomaly']==-1])}")
        st.write(f"Number of non anomalous values {len(data[data['anomaly']== 1])}")
        st.write(f'Total Number of Values: {len(data)}')

    #Concatenate df and df_final
    df_show = pd.concat([df, df_final], axis=1)

    #Run model
    if st.button("Run Outlier Detection"):
        # Preprocess data
        df_final = df_final.dropna()

        # Create model
        model_IF = IsolationForest(contamination=contamination, random_state=random_state)
        model_IF.fit(df_final[anomaly_inputs_final])

        # Predict anomalies
        df_final['anomaly_scores'] = model_IF.decision_function(df_final[anomaly_inputs_final])
        df_final['anomaly_scores'] = np.multiply(df_final['anomaly_scores'], 1)

        df_final['anomaly'] = model_IF.predict(df_final[anomaly_inputs_final])
        df_final['anomaly'] = np.multiply(df_final['anomaly'], 1)

        # Plot outliers
        #st.pyplot(outlier_plot(df_final, 'Isolation Forest', 'anomaly', 'anomaly_scores', [-2, 2], [-2, 2]))
        #st.pyplot(outlier_plot(df_final, 'Isolation Forest', 'Packet Size', 'anomaly_scores', [0, 0.8], [3, 1.5]))
        outlier_plot(df_final, "Isolation Forest")

        # Display anomalous data
        #df_show = pd.concat([df, df_final], axis=1)
        df_show = pd.concat([df.drop(columns=anomaly_inputs_final), df_final], axis=1)
        anomalous_df = df_show[df_show['anomaly'] == -1]

        ## Remove unnecessary columns
        #columns_to_remove = [
            #'Source IP Address_1', 'Source IP Address_2', 'Source IP Address_3', 'Source IP Address_4',
            #'Destination IP Address_1', 'Destination IP Address_2', 'Destination IP Address_3', 'Destination IP Address_4'
        #]
        #anomalous_df = anomalous_df.drop(columns=columns_to_remove, errors='ignore')
        anomalous_df = anomalous_df.T.drop_duplicates().T

        ## Convert int columns to str
        anomalous_df = anomalous_df.applymap(lambda x: str(x) if isinstance(x, int) else x)
        anomalous_df = anomalous_df.drop(["anomaly", "anomaly_scores"], axis=1)

        st.write(anomalous_df)

# Statistics Plotting =======================================================================================================================================
elif option == "Statistics Plotting":
    # Load data
    id = "1R7tyi0-l8Go1xsaKeyAQXNCOu3V-HwT7pJy_VgPPAGE"
    url = f'https://docs.google.com/spreadsheet/ccc?key={id}&output=xlsx'
    df = pd.read_excel(url, sheet_name=0)

    st.title("Outlier Detection Plotting")

    # Select columns
    df = df.drop(["Source IP Address", "Destination IP Address"], axis=1, errors='ignore')
    columns = df.columns.tolist()
    anomaly_inputs = st.multiselect("Select columns", columns, default=columns)

    id = "1R7tyi0-l8Go1xsaKeyAQXNCOu3V-HwT7pJy_VgPPAGE"
    url = f'https://docs.google.com/spreadsheet/ccc?key={id}&output=xlsx'
    df = pd.read_excel(url, sheet_name=0)

    # Preprocess IP address columns
    for col in anomaly_inputs[:]:
        if "IP" in col:
            df[col] = df[col].apply(lambda x: [int(i) for i in x.split('.')])
            new_cols = pd.DataFrame(df[col].tolist(), columns=[f"{col}_{i}" for i in range(1, 5)])
            df = pd.concat([df.drop(col, axis=1), new_cols], axis=1)
            anomaly_inputs.remove(col)
            anomaly_inputs.extend([f"{col}_{i}" for i in range(1, 5)])

    # Create df_final without string columns
    df_final = df.select_dtypes(exclude=['object'])
    anomaly_inputs_final = [col for col in anomaly_inputs if col in df_final.columns]

    # Model parameters
    contamination = st.slider("Contamination", min_value=0.01, max_value=0.5, step=0.01, value=0.1)
    random_state = st.slider("Random State", min_value=0, max_value=42, value=42)

    # Outlier plot function
    def outlier_plot(data, outlier_method_name, x_var, y_var, xaxis_limits=[0,1], yaxis_limits=[0,1]):
        print(f'Outlier Method: {outlier_method_name}')
        method = f'{outlier_method_name}_anomaly'
        st.write(f"Number of anomalous values {len(data[data['anomaly']==-1])}")
        st.write(f"Number of non anomalous values {len(data[data['anomaly']== 1])}")
        st.write(f'Total Number of Values: {len(data)}')

    # Run model
    if st.button("Run Outlier Detection Plotting"):
        # Preprocess data
        df_final = df_final.dropna()
        
        # Create model
        model_IF = IsolationForest(contamination=contamination, random_state=random_state)
        model_IF.fit(df_final[anomaly_inputs_final])
        
        # Predict anomalies
        df_final['anomaly_scores'] = model_IF.decision_function(df_final[anomaly_inputs_final])
        df_final['anomaly_scores'] = np.multiply(df_final['anomaly_scores'], 1)
        df_final['anomaly'] = model_IF.predict(df_final[anomaly_inputs_final])
        df_final['anomaly'] = np.multiply(df_final['anomaly'], 1)
        
        # Plot outliers
        df_show = pd.concat([df.drop(columns=anomaly_inputs_final), df_final], axis=1)
        anomalous_df = df_show[df_show['anomaly'] == -1]
        
        ## Remove unnecessary columns
        columns_to_remove = ['Source IP Address_1', 'Source IP Address_2', 'Source IP Address_3', 'Source IP Address_4', 
                            'Destination IP Address_1', 'Destination IP Address_2', 'Destination IP Address_3', 'Destination IP Address_4']
        anomalous_df = anomalous_df.drop(columns=columns_to_remove, errors='ignore', axis=1)
        anomalous_df = anomalous_df.T.drop_duplicates().T
        
        ## Convert int columns to str
        anomalous_df = anomalous_df.applymap(lambda x: str(x) if isinstance(x, int) else x)
        anomalous_df = anomalous_df.drop(["anomaly", "anomaly_scores"], axis=1)
        #anomalous_df = anomalous_df.drop([col for col in anomalous_df.columns if 'IP' in col or col in ["anomaly", "anomaly_scores"]], axis=1)
        
        # Function to count rows
        def count_rows(df, column, target_strings):
            counts = []
            for string in target_strings:
                count = df[column].str.contains(string, case=False).sum()
                counts.append({'Type': string, 'Count': count})
            return pd.DataFrame(counts)
        
        # Count rows
        columns = anomalous_df.columns.tolist()
        #plot_options = st.multiselect("Select plots", columns)
        plot_options = anomaly_inputs
        
        for plot in plot_options:
            target_strings = anomalous_df[plot].tolist()
            counts_df = count_rows(anomalous_df, plot, target_strings)
            st.subheader(f"{plot} Occurrences")
            st.bar_chart(counts_df, x='Type', y='Count')

# Automated Report =======================================================================================================================================
elif option == "Automated Report":
    apisheetskey = "1sIEI-_9N96ndRJgWDyl0iL65bACeGQ74MncOV4HQCXY"
    url_apikey = f'https://docs.google.com/spreadsheet/ccc?key={apisheetskey}&output=csv'
    df_apikey = pd.read_csv(url_apikey)

    platform = "Gemini"
    apikeyxloc = df_apikey['Platform'].str.contains(platform).idxmax()
    apikey = df_apikey.iloc[apikeyxloc, 2]

    genai.configure(api_key=apikey)

    # Create the model
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    )

    def report(contextsev) : 
        question = f"""You are an AI assistant which task is to create report based on data. 
        IN INDONESIAN LANGUAGE, Create at least four paragraphs of report based on the data of {contextsev}, 
        and always add space and enter in new line for every line, make the report as readable and tidy as possible. 
        DO NOT GIVE AN INTRO AND DO NOT SAY WHERE YOU GOT THE INFORMATION FROM, JUST GIVE THE ANSWER. 
        Use the statistics of the context to back up your analysis of the data, use every knowledge
        you have to make the report AMAZING!!!!"""
        chat_session = model.start_chat(history=[])
        questionpers = question
        response = chat_session.send_message(questionpers)
        answer = response.text
        answer = str(answer)
        st.text_area("Report", answer, height=850)
        #st.code(answer)

    # Load data
    id = "1R7tyi0-l8Go1xsaKeyAQXNCOu3V-HwT7pJy_VgPPAGE"
    url = f'https://docs.google.com/spreadsheet/ccc?key={id}&output=xlsx'
    df = pd.read_excel(url, sheet_name=0)

    st.title("Automated Report")

    # Select columns
    columns = df.columns.tolist()
    #anomaly_inputs = st.multiselect("Select columns", columns, default=columns)

    anomaly_inputs = st.selectbox('Choose part to create report', columns)
    anomaly_inputz = columns

    # Preprocess IP address columns
    for col in anomaly_inputz[:]:
        if "IP" in col:
            df[col] = df[col].apply(lambda x: [int(i) for i in x.split('.')])
            new_cols = pd.DataFrame(df[col].tolist(), columns=[f"{col}_{i}" for i in range(1, 5)])
            df = pd.concat([df.drop(col, axis=1), new_cols], axis=1)
            anomaly_inputz.remove(col)
            anomaly_inputz.extend([f"{col}_{i}" for i in range(1, 5)])

    # Create df_final without string columns
    df_final = df.select_dtypes(exclude=['object'])
    anomaly_inputs_final = [col for col in anomaly_inputz if col in df_final.columns]

    # Model parameters
    contamination = st.slider("Contamination", min_value=0.01, max_value=0.5, step=0.01, value=0.1)
    random_state = st.slider("Random State", min_value=0, max_value=42, value=42)

    # Outlier plot function
    def outlier_plot(data, outlier_method_name, x_var, y_var, xaxis_limits=[0,1], yaxis_limits=[0,1]):
        print(f'Outlier Method: {outlier_method_name}')
        method = f'{outlier_method_name}_anomaly'
        st.write(f"Number of anomalous values {len(data[data['anomaly']==-1])}")
        st.write(f"Number of non anomalous values {len(data[data['anomaly']== 1])}")
        st.write(f'Total Number of Values: {len(data)}')

    # Run model
    if st.button("Run Automated Report"):
        # Preprocess data
        df_final = df_final.dropna()
        
        # Create model
        model_IF = IsolationForest(contamination=contamination, random_state=random_state)
        model_IF.fit(df_final[anomaly_inputs_final])
        
        # Predict anomalies
        df_final['anomaly_scores'] = model_IF.decision_function(df_final[anomaly_inputs_final])
        df_final['anomaly_scores'] = np.multiply(df_final['anomaly_scores'], 1)
        df_final['anomaly'] = model_IF.predict(df_final[anomaly_inputs_final])
        df_final['anomaly'] = np.multiply(df_final['anomaly'], 1)
        
        # Plot outliers
        df_show = pd.concat([df.drop(columns=anomaly_inputs_final), df_final], axis=1)
        anomalous_df = df_show[df_show['anomaly'] == -1]
        
        ## Remove unnecessary columns
        columns_to_remove = ['Source IP Address_1', 'Source IP Address_2', 'Source IP Address_3', 'Source IP Address_4', 
                            'Destination IP Address_1', 'Destination IP Address_2', 'Destination IP Address_3', 'Destination IP Address_4']
        anomalous_df = anomalous_df.drop(columns=columns_to_remove, errors='ignore')
        anomalous_df = anomalous_df.T.drop_duplicates().T
        
        ## Convert int columns to str
        anomalous_df = anomalous_df.applymap(lambda x: str(x) if isinstance(x, int) else x)
        anomalous_df = anomalous_df.drop(["anomaly", "anomaly_scores"], axis=1)
        
        # Function to count rows
        def count_rows(df, column, target_strings):
            counts = []
            for string in target_strings:
                count = df[column].str.contains(string, case=False).sum()
                counts.append({'Type': string, 'Count': count})
            return pd.DataFrame(counts)
        
        # Count rows
        columns = anomalous_df.columns.tolist()
        #plot_options = st.multiselect("Select plots", columns)
        plot_options = [anomaly_inputs]
        
        for plot in plot_options:
            target_strings = anomalous_df[plot].tolist()
            counts_df = count_rows(anomalous_df, plot, target_strings)
            st.subheader(f"{plot} Occurrences")
            st.bar_chart(counts_df, x='Type', y='Count')
            str_df = counts_df.to_string()
            st.header(f"{plot} Statistics Report")
            report(str_df)
            #time.sleep(30)