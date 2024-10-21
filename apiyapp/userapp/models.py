from django.db import models

class User(models.Model):
    #pkid=models.IntegerField(unique=True,primary_key=True)
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=100)
    gender = models.CharField(null=True, max_length=20, default="not specified")
    Age = models.IntegerField(null=True, default=5)  
    bio = models.CharField(null=True, default="hi", max_length=255)  # Set a max_length for bio

class Apparel(models.Model):
    #pkid=models.IntegerField(unique=True,primary_key=True)
    ownership = models.ForeignKey(User, on_delete=models.CASCADE)
    color = models.CharField(max_length=50)
    material = models.CharField(max_length=50)
    occasion = models.CharField(max_length=20, choices=(
        ("CASUAL", "Casual"),
        ("TRADITIONAL", "Traditional"),
        ("FORMAL", "Formal"),
    ), default="CASUAL")
    upper_lower = models.CharField(max_length=50)  # e.g., "upper" or "lower"
    type = models.CharField(max_length=50)  # e.g., "shirt", "pants"
    image = models.ImageField(upload_to='apparels/', blank=True, null=True)  # Use ImageField for storing apparel images
    def __str__(self):
        return f"{self.occasion} {self.type} {self.upper_lower} {self.color} - {self.ownership.name}"


class Designs(models.Model):
    #pkid=models.IntegerField(unique=True,primary_key=True)
    color = models.CharField(max_length=50, default="black")
    base_apparel = models.ForeignKey(Apparel, on_delete=models.CASCADE)
    design_image = models.ImageField(upload_to='designs/', blank=True, null=True)  # Store design image

class Shoes(models.Model):
    #pkid=models.IntegerField(unique=True,primary_key=True)
    type = models.CharField(max_length=50)
    color = models.CharField(max_length=50)
    shoe_image = models.ImageField(upload_to='shoes/', blank=True, null=True)  # Store shoe image

class UpperAccessories(models.Model):  # Corrected to UpperAccessories for consistency
    #pkid=models.IntegerField(unique=True,primary_key=True)    
    label = models.CharField(max_length=50)
    color = models.CharField(max_length=50)
    accessory_image = models.ImageField(upload_to='upper_accessories/', blank=True, null=True)  # Store accessory image


class Combination(models.Model):
    #pkid=models.IntegerField(unique=True,primary_key=True)
    ownership = models.ForeignKey(User, on_delete=models.CASCADE)
    apparels = models.ManyToManyField(Apparel)  # Store multiple apparel items
    upperaccessories=models.ManyToManyField(UpperAccessories)
    shoes=models.ForeignKey(Shoes,on_delete=models.CASCADE,null=True,blank=True)
    occasion = models.CharField(max_length=20, choices=(
        ("CASUAL", "Casual"),
        ("TRADITIONAL", "Traditional"),
        ("FORMAL", "Formal"),
    ), default="CASUAL")

    image = models.ImageField(upload_to='combinations/', blank=True, null=True)  # Store combination image


class Accessories(models.Model):
    #pkid=models.IntegerField(unique=True,primary_key=True)
    attire = models.ForeignKey(Combination, on_delete=models.CASCADE)
    type = models.CharField(max_length=50)
    shoes = models.OneToOneField(Shoes, on_delete=models.CASCADE, null=True, blank=True)
    upperAccessories = models.OneToOneField(UpperAccessories, on_delete=models.CASCADE, null=True, blank=True)


class Recommendation(models.Model):
    #pkid=models.IntegerField(unique=True,primary_key=True)
    ownership = models.ForeignKey(User, on_delete=models.CASCADE)
    acceptance = models.IntegerField(default=0)
    remarks = models.CharField(max_length=255, default="cool")  # Added max_length
    combination_id=models.ForeignKey(Combination,on_delete=models.CASCADE,default=0)
    upperaccessories=models.ManyToManyField(UpperAccessories)
    shoes=models.ForeignKey(Shoes,on_delete=models.CASCADE,null=True,blank=True)

class Post(models.Model):
    #pkid=models.IntegerField(unique=True,primary_key=True)
    ownership = models.ForeignKey(User, on_delete=models.CASCADE)
    combination=models.ForeignKey(Combination,on_delete=models.CASCADE)
    likes = models.IntegerField(default=0)
    date_posted = models.DateField(auto_now_add=True)