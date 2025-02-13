## Project title: Exploring HESA Data with Large Language Models for Dynamic Visualisation

### Student name: Vladimirs Ribakovs

### Student email: vr112@student.le.ac.uk

### Project description: 
The Higher Education Statistics Agency (HESA) collects and provides open-source data on UK universities. They are offering insights into institutional performance, funding decisions and benchmarking. The aim of this project is to create a dynamic and user-friendly dashboard using large language models (LLM) and make it allow users to analyse and visualise HESA's data.

The dashboard will enable users to input queries in natural language. The LLM will interpret and convert these queries into structured analyses. These analyses will provide meaningful insights through tables, charts, and comparative reports. The project will initially focus on implementing a static dashboard consisting of the most recent data up to March 2025. The future goal is to make the dashboard update automatically when there is new data provided by HESA.

This project integrates data analysis, natural language processing (NLP), and interactive visualisation. This allows universities to make it easier to understand their positioning within the sector and to compare themselves against peer institutions. It can help to derive strategic insights from HESA’s extensive datasets.

### List of requirements (objectives): 

Essential:
- Natural Language Query Interpretation: Users will type questions and the LLM-powered system will interpret them to retrieve and process relevant HESA data.
- Integrating and Processing Data: HESA open-source data needs to be extracted, cleaned and well-structured so they can be analysed.
- Interactive Dashboard: A user-friendly interface that dynamically displays results. It can display results in different ways like tables, charts or summary reports
- Comparative Analysis: Ability to compare universities, group them by Mission Groups (Russell Group, University Alliance, etc.), and benchmark them based on relevant metrics.
- Data Export Capabilities: Users should be able to download reports in a structured format (CSV, PDF, or Excel) for further use.

Desirable:
- Automated Data Retrieval: The system should be able to fetch and update HESA data automatically. This will eliminate the need for manual data downloads and updates.
- AI Analysis: The LLM will not just fetch data but also provide insights and contextual analysis about it. It will identify trends and make recommendations based on historical data.
- Advanced Visualisations: It will support more advanced visualisation techniques like interactive graphs, trend projections, and geographic mappings.
- Custom Query-Building Interface: For users who are unfamiliar with free-text queries, a quick guide will be provided suggesting relevant questions based on available datasets.
- Historical Data Tracking: Ability to track changes in university performance over time and highlight key shifts and trends.
- Caching for Frequent Queries: Implement a way to cache frequently requested queries and store them to make the next similar query more efficient and faster.
- LLM Interpretation Feedback: Show users how the system interpreted their query (showing an English-like readable query).
- Queries Logs and Protection: Log out users' queries and LLM outputs to improve future prompts. Create a protection mechanism for enormous queries so they will not break the system.

Optional:
- Predictive Modelling: Using historical HESA data and machine learning models to predict future university performance metrics.
- Chatbot Integration: Embedding an AI chatbot within the dashboard to allow users to converse with the data and refine their queries dynamically.
- API for External Use: Develope an API that will allows third parties applications to integrate HESA data analysis functionalities.
- Multi-User Concurrency: Allows users to provide multiple queries at the same time using async job queues (like Celery) and session management. 
- Collect feedback: Receive feedback from users and use it to update the UI, query handling and LLM prompts.
- CSV File Validation: Create sanitiser that validates files, scanning for size anomalies and parsing CSVs as plain text.
- Querying Tutorial: Show end users how the querying works and provide suggestions on how they can write their first query.
- Scalability & Advanced Data Pipelines: If data grows to an extensive amount consider moving from CSV files to a database with indexing (like PostgreSQL) and creating advanced data pipelines.


## Information about this repository
This is the repository that you are going to use **individually** for developing your project. Please use the resources provided in the module to learn about **plagiarism** and how plagiarism awareness can foster your learning.

Regarding the use of this repository, once a feature (or part of it) is developed and **working** or parts of your system are integrated and **working**, define a commit and push it to the remote repository. You may find yourself making a commit after a productive hour of work (or even after 20 minutes!), for example. Choose commit message wisely and be concise.

Please choose the structure of the contents of this repository that suits the needs of your project but do indicate in this file where the main software artefacts are located.
