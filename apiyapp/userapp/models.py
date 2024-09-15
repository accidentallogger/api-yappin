from django.db import models

class User(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=100)
    gender = models.CharField(null=True,max_length=20,default="not specified")
    Age=models.IntegerField(null=True,default="5")
    bio=models.CharField(null=True,default="hi")

class Apparel(models.Model):
    ownership = models.ForeignKey(User, on_delete=models.CASCADE)
    color = models.CharField(max_length=50)
    material = models.CharField(max_length=50)
    upper_lower = models.CharField(max_length=50)  # e.g., "upper" or "lower"
    type = models.CharField(max_length=50)  # e.g., "shirt", "pants"
    image = models.BinaryField()
