## Project title: Exploring HESA Data with Large Language Models for Dynamic Visualisation

### Student name: Vladimirs Ribakovs

### Student email: vr112@student.le.ac.uk

### Project description: 

The Higher Education Statistics Agency (HESA) collects and provides open-source university data around the UK. They offer information about institutional performance, funding decisions, and benchmarking. This project aims to create a dynamic and user-friendly dashboard using large language models (LLM). This dashboard is expected to allow users to analyse and visualise HESA's data.

The dashboard will enable users to input queries in natural language. The LLM will interpret and convert these queries into structured analyses, which will provide meaningful information through tables, charts, and reports. The project will initially focus on implementing a static dashboard consisting of the most recent data up to March 2025. The future goal is to make the dashboard to automatically update when the new data is provided by HESA.

This project integrates data analysis, natural language processing (NLP), and interactive visualisation. This makes universities to easier understand their positioning within the sector and to compare themselves against other institutions. It can help to derive strategic insights from HESA’s extensive datasets.


### List of requirements (objectives): 


Essential:

- Natural Language Query Interpretation: Users will type questions and the LLM-powered system will interpret them to retrieve and process relevant HESA data.

- Integrating and Processing Data: HESA open-source data needs to be extracted, cleaned and be well-structured so it can be analysed.

- Interactive Dashboard: A user-friendly interface that dynamically displays results. It can display results in different ways like tables, charts or summary reports.

- Comparative Analysis: Comparing universities by grouping them by Mission Groups (Russell Group, University Alliance, etc.), and benchmarking them based on relevant metrics.

- Data Export Capabilities: Users should be able to download reports in formats like (CSV, PDF, or Excel) for further use.


Desirable:

- Automated Data Retrieval: The system should be able to fetch and update HESA data automatically. This will eliminate the need for manual data downloads and updates.

- Using AI To Improve Analysis: The LLM not only fetches data but also provides analysis about it. It identifies trends and comes up with recommendations based on historical data.

- Custom Query Building: Users who are unfamiliar with free-text queries will be provided with a quick guide to write relevant questions based on available datasets.

- Advanced Visualisation: Users can view data using advanced visualisation techniques like interactive graphs and trend projection.

- Historical Data Tracking: Keep track of changes in university performance over time and highlight important shifts and trends.

- Caching for Frequent Queries: Implement a way to cache frequently requested queries and store them to make the next similar query more efficient and faster.

- LLM Interpretation Feedback: Show users how the system interpreted their query (showing an English-like readable query).

- Queries Logs and Protection: Log out users' queries and LLM outputs to improve future prompts. Create a protection mechanism so that big queries will not break the system.


Optional:

- Predictive Modelling: Using historical HESA data and machine learning models to predict future university metrics.

- Chatbot Integration: Embedding an AI chatbot within the dashboard to allow users to converse with the data and refine their queries dynamically.

- API for External Use: Develop an API that will allow external applications to integrate HESA data analysis functionalities.

- Multi-User Concurrency: Allows users to provide multiple queries at the same time using asynchronous job queues (like Celery) and session management.

- Collect feedback: Receive feedback from users and use it to update the UI to make it more user-friendly. And to improve query handling and LLM prompts.

- CSV File Validation: Create a sanitiser that validates files, scans for size anomalies and parses CSVs as plain text to prevent system breaking.


## informationrmation about this repository
This is the repository that you are going to use **individually** for developing your project. Please use the resources provided in the module to learn about **plagiarism** and how plagiarism awareness can foster your learning.

Regarding the use of this repository, once a feature (or part of it) is developed and **working** or parts of your system are integrated and **working**, define a commit and push it to the remote repository. You may find yourself making a commit after a productive hour of work (or even after 20 minutes!), for example. Choose commit message wisely and be concise.

Please choose the structure of the contents of this repository that suits the needs of your project but do indicate in this file where the main software artefacts are located.
