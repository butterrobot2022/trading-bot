from django.shortcuts import render
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


def join_waitlist(request, email):
    signups = SignUp.objects.all()
    print(signups)
    return

// Models Code
from django.db import models
import datetime
# Create your models here.

class SignUp(models.Model):
    email = models.EmailField(unique=True)
    timestamp = models.DateTimeField(default=datetime.datetime.now())


    def __str__(self):
        return f'New SignUp email is {email}'

// Html Code
<!DOCTYPE html>

<head>
    <title>Interface</title>
    <link rel="stylesheet" href="/static/horizonAI/styles.css">  
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-rbsA2VBKQhggwzxH7pPCaAqO46MgnOM80zW1RWuH61DGLwZJEdK2Kadq2F9CUG65" crossorigin="anonymous"> 
</head>

<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container-fluid">
          <a class="navbar-brand" href="#">horizon.ai</a>
          <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
            <span class="navbar-toggler-icon"></span>
          </button>
          <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav">
              <!-- <li class="nav-item">
                <a class="nav-link active" aria-current="page" href="#">Home</a>
              </li> -->
              <li class="nav-item">
                <a class="nav-link" href="#" style="color: white;">Sign Up</a>
              </li>
              <!-- <li class="nav-item">
                <a class="nav-link" href="#">Get started today</a>
              </li> -->

            </ul>
            <a href="{% url 'horizonAI:dictionary' %}"><button class="btn btn-primary">Get started today</button></a>
          </div>
        </div>
      </nav>
      <div class="waitlist-div">
      <h4>Automate your trading workflow and achieve consistent gains</h4>
      <p class="explanation">Trading in the financial markets is a high stakes game. Or is it? We help you navigate the financial markets and become a
        consistent winner with the help of an AI assistant.
      </p>
      <form class="waitlist-email">

        <div class="form-group" style="display: flex; gap: 10px; margin-top: 30px;">
            <input type="email" class="form-control waitlist-email-input" id="exampleInputEmail1" aria-describedby="emailHelp" placeholder="Enter Email">
            <!-- <small id="emailHelp" ></small> -->
            <button class="btn btn-primary join-waitlist-button">Join Waitlist</button>
        </div>
      </form>
    </div>
</body>

</html>