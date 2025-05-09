Ensure that the following requirements are used with the right version:
- Python 3.10+ installed
- Node.js and npm installed

! Please wait until any command is fully executed, it may take time !
! DO NOT CANCEL THEM (ctrl + c) !

1. Clone the Repository or Use submitted on BB
git clone https://campus.cs.le.ac.uk/gitlab/pgt_project/24_25_spring/vr112.git
or just access the submitted folder


2. Create and Activate a Virtual Environment

Before typing, please be in the root level of the project: vr112\hesa-llm-visualisation

Type the following commands (based on your Operating System)

For Windows:
python -m venv venv
venv\Scripts\activate

For macOS/Linux:
python -m venv venv
source venv/bin/activate


3. Install Python Dependencies
Download dependencies:
Note: this command takes about 5 minutes, depending on what is already downloaded on the system, internet speed and other criteria.
Please be patient and wait for similar messages that would indicate that the commands have executed
```[notice] A new release of pip is available: 24.0 -> 25.1.1
```
```[notice] To update, run: python.exe -m pip install --upgrade pip
```
pip install -r requirements.txt

If facing issue, try creating `requirements.txt` file again:
pip freeze > requirements.txt

And then typing again:
pip install -r requirements.txt

Please make sure you are located at this level:
vr112\hesa-llm-visualisation\


4. Install JavaScript Dependencies
Please make sure you are located at this level:
vr112\hesa-llm-visualisation\ otherwise type cd hesa-llm-visualisation 

When in the right location type:
npm install


5. APY Key

Please do not share or leak the provided API Key. In case of any issue or inquiries regarding API Key, please contact the owner of the repo, aka me (vr112)


6. Prepare the data
Curretly there is no provided cleaned csv file because it takes more space to store it, so you need to clean the raw provided data by typing:
py .\csv_cleaning.py

Please make sure you are located at this level:
vr112\hesa-llm-visualisation\ otherwise type cd hesa-llm-visualisation 


7. Run the Django Development Server

Please make sure you are located at this level:
vr112\hesa-llm-visualisation\ otherwise type cd hesa-llm-visualisation 

When in the right directory type:
python manage.py runserver

Alternatively, you can use the provided batch file:
.\run_django_server.bat

The server will run at http://127.0.0.1:8000/




Running Tests

Please make sure you are located at this level:
vr112\hesa-llm-visualisation\ otherwise type cd hesa-llm-visualisation 

Then type:
tests\scripts\run_all_tests.bat

This will execute all unit and integration tests and display the results.
