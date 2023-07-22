from django.shortcuts import render
from django.http import JsonResponse
from json import loads
from .models import SignUp
import datetime
# Create your views here.


def home(request):
    signups = SignUp.objects.all()
    print(signups)
    return render(request, 'horizonAI/home.html')


def dictionary(request):
    return render(request, 'horizonAI/dictionary.html')


def books(request):
    return render(request, 'horizonAI/books.html')


def journal(request):
    return render(request, 'horizonAI/journal.html')


def join_waitlist(request):
    if request.method == 'POST':
        body = loads(request.body)
        email = body.get('email') # Get the email from the AJAX request
        print(email)
        if email:
            # Check if the email already exists in the database
            if not SignUp.objects.filter(email=email).exists():
                signup = SignUp(email=email)
                signup.save()
                return JsonResponse({'message': 'Successfully added email!'})
            else:
                return JsonResponse({'message': 'Email already exists'})

    # Return a failure JSON response if the email is not provided or it's not a POST request
    return JsonResponse({'message': 'Invalid request'}, status=400)
