import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from google.cloud import bigquery
from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_google_vertexai import VertexAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from google.cloud import aiplatform
from langchain_community.document_loaders import DataFrameLoader
import os
import json
import tempfile


# Show title and description.
st.title("💬 Tech Recruiter Chatbot")
st.write(
    "This is a simple chatbot that uses Gemini model to generate responses. "
    "To use this app, you need to provide an Gemini API key, which you can get [here](https://aistudio.google.com/app/apikey). "
)
if st.button('Clear', type="primary"):
    st.session_state.messages.clear()



# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management

def gemini_model(google_api_key):
  
    try:
        model = ChatGoogleGenerativeAI(model='gemini-1.5-flash-002', temperature=0, google_api_key=google_api_key)
    except Exception as e:
        st.error(f"An error occurred while setting up the Gemini model: {e}")

    return model

def app(model, query: str):

    table_schema = f"""
    table_metadata = assignment_table
    table_name: Applicant_Details,
    description: This table contains detailed records related to the assignment and application process. It links applications, recruiters, jobs, candidates, and associated statuses. Each record represents a unique combination of applications, jobs, and recruiters with additional metadata about candidates, jobs, and application processes.
        data_date:
            data_type: Datetime,
            description: Date of the application, at least 1 week before corresponding assignment dates.,
            example_value: 2023-10-11
            ,
        appli_id:
            data_type: String,
            description: Unique identifier for each application.,
            example_value: APP1301
            ,
        status_id:
            data_type: String,
            description: Current status of recruiting process judge by recruiter (B01=reviewing, B02=proceed, B03 reject),
            example_value: A02
            ,
        status_name:
            data_type: String,
            description: Descriptive name of the status, representing various stages in a process e.g., Review, Interview, Test, Signed, Reject,
            example_value: Interview
            ,
        appli_date:
            data_type: Unknown,
            description: No description available.,
            example_value: 2023-09-14
            ,
        rec_id:
            data_type: String,
            description: Identifier for the recruiter handling the application.,
            example_value: R001
            ,
        rec_fullname:
            data_type: Unknown,
            description: No description available.,
            example_value: John Doe
            ,
        rec_phone_number:
            data_type: Unknown,
            description: No description available.,
            example_value: 081-123-4567
            ,
        rec_department:
            data_type: Unknown,
            description: No description available.,
            example_value: Recruitment IT and Software Development
            ,
        rec_salary:
            data_type: integer,
            description: Salary of each recruiter,
            example_value: 46900.0
            ,
        channel_name:
            data_type: Unknown,
            description: No description available.,
            example_value: LinkedIn
            ,
        cust_name:
            data_type: Unknown,
            description: No description available.,
            example_value: SupportLife
            ,
        cust_industry_name:
            data_type: Unknown,
            description: No description available.,
            example_value: Non-Profit
            ,
        job_id:
            data_type: String,
            description: Identifier for the job applied, links to Job table. Same Job_ID maps to the same Cust_ID.,
            example_value: JOB1714
            ,
        job_budget:
            data_type: Unknown,
            description: No description available.,
            example_value: 110000.0
            ,
        job_title_name:
            data_type: String,
            description: Job Name eg. Data Analyst, Data Engineer,
            example_value: Frontend Developer
            ,
        job_skill_name:
            data_type: Unknown,
            description: No description available.,
            example_value: ['Cyber Threat Intelligence' 'Malware Analysis' 'Penetration Testing'
                            'Threat Analysis' 'Financial Modeling' 'Financial Planning'
                            'Investment Strategies' 'Risk Analysis' 'HTML/CSS' 'JavaScript'
                            'Responsive Design' 'UI Frameworks']
            ,
        job_deadline_dt:
            data_type: Datetime,
            description: Last date proposed with customer to fill this job position and get commission,
            example_value: 2023-12-02
            ,
        can_id:
            data_type: String,
            description: Identifier for the candidate who applied, links to the Candidate table.,
            example_value: CAN00005
            ,
        can_fullname:
            data_type: Unknown,
            description: No description available.,
            example_value: Holly Sanchez
            ,
        can_phone:
            data_type: Unknown,
            description: No description available.,
            example_value: 080-4632-7736
            ,
        can_email:
            data_type: Unknown,
            description: No description available.,
            example_value: holly.sanchez@example.com
            ,
        can_current_pos_id:
            data_type: Unknown,
            description: No description available.,
            example_value: JOBT004
            ,
        can_experience_year:
            data_type: Unknown,
            description: No description available.,
            example_value: 2
            ,
        can_current_sar:
            data_type: Unknown,
            description: No description available.,
            example_value: 71500
            ,
        can_toeic_score:
            data_type: Unknown,
            description: No description available.,
            example_value: 634
            ,
        can_project_exp:
            data_type: Unknown,
            description: No description available.,
            example_value: Budgeting Tool Design
            ,
        can_skill_name:
            data_type: Unknown,
            description: No description available.,
            example_value: ['Communication Skills' 'Python' 'System Design'
                            'Stakeholder Communication' 'Budget Management' 'Financial Planning']
    """

    big_query_prompt = """
    You are a sophisticated BigQuery SQL query generator.
    Translate the following natural language request (human query) into a valid BigQuery syntax (SQL query).
    Consider the table schema provided.
    FROM always `psyched-camp-375615.madt8102_db.assignment_map_all_oct_dec_23`
    Format the SQL Query result as JSON with 'big_query' as a key.

    ###
    Example:
    Table Schema:
    table_name: Applicant_Details,
    description: 'This table contains detailed information about job applications, including application IDs, associated job IDs, recruiter details, customer information, application status, expected salary, and the CV file location.",
    columns:
        Appli_ID:
            data_type': string,
            description': Application ID of the applicant.,
            example_value': APP0263
        Job_ID':
            data_type': string,
            description': Job ID associated with the application.,
            example_value': JOB1678

    Human Query: Ranking the popular job from most to least popular

    SQL Query: SELECT Job_ID, COUNT(*) AS ApplicationCount
    FROM `madt8102-test-pipeline-442401.hr_management_dataset.application_table`
    GROUP BY Job_ID
    ORDER BY ApplicationCount DESC;

    ###
    Table Schema: {table_schema}
    Human Query: {query}
    SQL Query:
    """

    response_prompt = """
    Summary the information you get and answer the question, using question and query result to answer back to user

    ###
    Example:
    Question: What is the most popular channel for candidate?
    Query result: SELECT Channel_ID, COUNT(*) AS ApplicationCount FROM `madt8102-test-pipeline-442401.hr_management_dataset.application_table` GROUP BY Channel_ID ORDER BY ApplicationCount DESC LIMIT 1
    Answer: The most popular channel for candidate is CHN_01 which is 1088 candidates.

    ###
    Question: {user_query}
    Query result: {sql_result}
    Answer:
    """

    # dataset_id = 'hr_management_dataset'
    # table_id = 'application_table'
    project_id = "psyched-camp-375615"
    client = bigquery.Client(project=project_id)

    parser = JsonOutputParser()
    bigquery_prompt_template = PromptTemplate(template=big_query_prompt, input_variables=['table_schema', 'query'])
    bigquery_chain = bigquery_prompt_template | model | parser
    sql_bigquery_result = bigquery_chain.invoke({"table_schema": table_schema, "query": query})
    bigquery_query = sql_bigquery_result['big_query']
    # print(bigquery_query)
    bigquery_query_result = client.query(bigquery_query).to_dataframe()

    response_prompt_template = PromptTemplate(template=response_prompt, input_variables=['user_query', 'sql_result'])
    response_chain = response_prompt_template | model
    response_result = response_chain.invoke({"user_query": query, "sql_result": bigquery_query_result})

    return response_result.content, bigquery_query

def can(openai_api_key):

    # dataset_id = 'hr_management_dataset'
    # table_id = 'application_table'
    # project_id = 'madt8102-test-pipeline-442401'
    # bucket = "madt8102_hr_management_vector_bucket"
    # bucket_uri = f"gs://{bucket}"
    # region = "us-central1"

    # aiplatform.init(project=project_id, location=region, staging_bucket=bucket_uri)
    # embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)
    vector_store = InMemoryVectorStore(embeddings)

    return vector_store

def can_sum(model, skill_need: str, sql_result):

    can_sum_prompt = """
    You are the expert recruiter.
    Summary and suggestion using the information you get.
    The objective of the answer is to summary the candidate information for recruiter 
    it should be included name, last name, phone, email, currect salary, project experience, and candidate skill name.
    Format as text bullet point and add new line every point to make it easy to read for example
    Name:
    Last Name:
    Phone:
    Email:
    Current Salary:
    Experience:
    Skill:
    --------------
    
    Recommendation:

    ###
    Skill need: {skill_need}
    Query result: {sql_result}
    Answer:
    """

    can_prompt_template = PromptTemplate(template=can_sum_prompt, input_variables=['user_query', 'sql_result'])
    can_chain = can_prompt_template | model
    can_result = can_chain.invoke({"skill_need": skill_need, "sql_result": sql_result})

    return can_result

def main():
    with st.sidebar:
        st.title(":red[Credential and Key]")
        uploaded_file = st.file_uploader("Upload Credential File .json", type="json")

        if not uploaded_file:
            st.info("Please add your Bigquery creditial to continue.", icon="🗝️")

        if uploaded_file is not None:
            if st.button("Add Creditial"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
                        temp_file.write(uploaded_file.read())
                        temp_file_path = temp_file.name

                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path
                st.success("Bigquery creditial successfully uploaded.", icon="✅")

        google_api_key = st.text_input("Gemini API Key", type="password")

        if not google_api_key:
            st.info("Please add your Gemini API key and Bigquery creditial to continue.", icon="🗝️")
            
        if google_api_key is not None:
            if st.button("Add Gemini API Key"):
                st.success("Gemini API key successfully uploaded.", icon="✅")

        openai_api_key = st.text_input("OpenAI API Key", type="password")

        if not openai_api_key:
            st.info("Please add your OpenAI API key and Bigquery creditial to continue.", icon="🗝️")
            
        if openai_api_key is not None:
            if st.button("Add OpenAI API Key"):
                st.success("OpenAI API key successfully uploaded.", icon="✅")

        with st.sidebar:
            st.subheader(":blue[SQL Syntax]")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["content"])
    tab1, tab2  = st.tabs(["Assignment", "Candidate Matching"])
    with tab1:
        try:
            if user_input := st.chat_input("What is up?"):

                # with st.chat_message("user"):
                #     st.markdown(user_input)

                st.session_state.messages.append({"role": "user", "content": user_input})

                model = gemini_model(google_api_key=google_api_key)
                bot_response, bigquery_query = app(model, query=user_input)

                # with st.chat_message("assistant"):
                #     with st.spinner("Thinking..."):
                #         st.markdown(bot_response)

                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                with st.sidebar:
                    st.code(bigquery_query)

            chat_css = """
            <style>
            .chat-container {
                display: flex;
                align-items: flex-start;
                margin: 10px 0;
            }
            .user-message {
                margin-right: auto;
                background-color: #fce4ec;
                color: black;
                padding: 10px;
                border-radius: 10px;
                max-width: 70%;
            }
            .assistant-message {
                margin-left: auto;
                background-color: #fff9c4;
                color: black;
                padding: 10px;
                border-radius: 10px;
                max-width: 70%;
            }
            .icon {
                display: flex;
                align-items: center;
                justify-content: center;
                width: 40px;
                height: 40px;
                background-color: #f5f5f5;
                border-radius: 50%;
                font-size: 20px;
                margin: 0 10px;
            }
            .user-container {
                display: flex;
                flex-direction: row-reverse;
                align-items: center;
            }
            .assistant-container {
                display: flex;
                flex-direction: row;
                align-items: center;
            }
            </style>
            """
            st.markdown(chat_css, unsafe_allow_html=True)
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(
                        f"""
                        <div class="chat-container user-container">
                            <div class="user-message">{message['content']}</div>
                            <div class="icon">👤</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="chat-container assistant-container">
                            <div class="icon">🤖</div>
                            <div class="assistant-message">{message['content']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        except:
            st.markdown("🗝️Please make sure you have already added Bigquery creditial and Gemini API key to continue.")

    with tab2:
        if user_input := st.chat_input("Type here ..."):

            vector_store = can(openai_api_key=openai_api_key)

            project_id = "psyched-camp-375615"
            client = bigquery.Client(project = project_id)
            query = """
            select *
                from psyched-camp-375615.madt8102_db.assignment_map_all_oct_dec_23

            """
            chk = client.query(query).to_dataframe()
            df = (chk.sort_values(by='data_date', ascending=False)
                    .drop_duplicates(subset='can_fullname', keep='first')
                    .drop_duplicates(subset='can_skill_name', keep='first'))
            print(chk.shape)
            df['can_skill_name_1'] = df['can_skill_name'].astype(str)
            df['can_skill_name_1'] = df['can_skill_name_1'].str.replace('[', '').str.replace(']', '')

            loader = DataFrameLoader(data_frame=df, page_content_column="can_skill_name_1")
            docs = loader.load()

            vector_store.add_documents(documents=docs)
            search = vector_store.similarity_search(user_input, k=1)
            search_result = search[0].metadata

            model = gemini_model(google_api_key=google_api_key)
            bot_response = can_sum(model, skill_need=user_input, sql_result=search_result)

            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": bot_response.content})

            chat_css = """
            <style>
            .chat-container {
                display: flex;
                align-items: flex-start;
                margin: 10px 0;
            }
            .user-message {
                margin-right: auto;
                background-color: #fce4ec;
                color: black;
                padding: 10px;
                border-radius: 10px;
                max-width: 70%;
            }
            .assistant-message {
                margin-left: auto;
                background-color: #fff9c4;
                color: black;
                padding: 10px;
                border-radius: 10px;
                max-width: 70%;
            }
            .icon {
                display: flex;
                align-items: center;
                justify-content: center;
                width: 40px;
                height: 40px;
                background-color: #f5f5f5;
                border-radius: 50%;
                font-size: 20px;
                margin: 0 10px;
            }
            .user-container {
                display: flex;
                flex-direction: row-reverse;
                align-items: center;
            }
            .assistant-container {
                display: flex;
                flex-direction: row;
                align-items: center;
            }
            </style>
            """
            st.markdown(chat_css, unsafe_allow_html=True)
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(
                        f"""
                        <div class="chat-container user-container">
                            <div class="user-message">{message['content']}</div>
                            <div class="icon">👤</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="chat-container assistant-container">
                            <div class="icon">🤖</div>
                            <div class="assistant-message">{message['content']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

if __name__ == '__main__':
    main()