from ast import alias
from concurrent.futures import process
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages

import twitter

from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import pandas as pd


# Create your views here.

def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):

    return render(request, 'users/UserHomePage.html', {})



    
def DatasetView(request):
    path = settings.MEDIA_ROOT + "//" + 'twitter.csv'
    df = pd.read_csv(path, nrows=100)
    df = df.to_html
    return render(request, 'users/viewdataset.html', {'data': df})



def usrtwitterFNDML(request):
    from .utility import twitterMLEDA
    svm_acc, svm_report = twitterMLEDA.process_SVM()
    svm_report = pd.DataFrame(svm_report).transpose()
    svm_report = pd.DataFrame(svm_report)
  
    nb_acc, nb_report = twitterMLEDA.process_naiveBayes()
    nb_report = pd.DataFrame(nb_report).transpose()
    nb_report = pd.DataFrame(nb_report)
   

    return render(request, 'users/twitterMl.html',
                  {
                    'svm_report': svm_report.to_html, 'svm_acc': svm_acc,
                     'nb_report': nb_report.to_html, 'nb_acc': nb_acc,
                      
                })    



def predictTrustWorthy(request):
    if request.method == 'POST':
        test_user_data  = request.POST.get('news')
        print(test_user_data)
        from .utility import twitterMLEDA
        result = twitterMLEDA.fake_news_det(test_user_data)
        return render(request, 'users/testform.html', {'msg': result})
    else:
        return render(request, 'users/testform.html', {})                



  













      











