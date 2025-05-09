

Ensure that the following requirements are used with the right version:
- Python 3.10+ installed
- Node.js and npm installed


1. Clone the Repository or Use sumibted on BB
git clone https://campus.cs.le.ac.uk/gitlab/pgt_project/24_25_spring/vr112.git
or just access the sumbited folder


2. Create and Activate a Virtual Environment

Before typing please be in the root level on the project: vr112\hesa-llm-visualisation

For Windows:
Type following commands
python -m venv venv
venv\Scripts\activate

For macOS/Linux:
python -m venv venv
source venv/bin/activate


3. Install Python Dependencies
Download dependencies:
pip install -r requirements.txt

It facing issue try create `requirements.txt` file again:
pip freeze > requirements.txt

And then typing again:
pip install -r requirements.txt

Please make sure you located at this level:
vr112\hesa-llm-visualisation\


4. Install JavaScript Dependencies
Please make sure you located at this level:
vr112\hesa-llm-visualisation\ otherwise type cd hesa-llm-visualisation 

When in the right location type:
npm install


5. Configure APY Key

Please do not share or leak the provide API Key. In case of any issue or inqueries reagrding API Key please contanct the own of the repo aka me (vr112)

Option 1:
1. Create data.env file in this directory -> vr112\hesa-llm-visualisation\
2. Put the API key as provided in the BB submision

Option 2:
1. Just drag the provided seperatly sumibted data.env file in the directory -> vr112\hesa-llm-visualisation\


6. Run the Django Development Server

Please make sure you located at this level:
vr112\hesa-llm-visualisation\ otherwise type cd hesa-llm-visualisation 

When in the right directory type:
python manage.py runserver

Alternatively, you can use the provided batch file:
.\run_django_server.bat

The server will run at http://127.0.0.1:8000/




Running Tests

Please make sure you located at this level:
vr112\hesa-llm-visualisation\ otherwise type cd hesa-llm-visualisation 

Then type:
tests\scripts\run_all_tests.bat

This will execute all unit and integration tests and display the results.
