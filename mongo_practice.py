# from collections.abc import MutableMapping
from pymongo import MongoClient
cluster = MongoClient('mongodb+srv://lakshmishreea2003:Chintu1234567890@cluster0.aln4gyt.mongodb.net/?retryWrites=true&w=majority',connect =False)

db = cluster['data-silent-bridge']
collection = db['practice']

result = collection.find_one({"user_name": "Soumi"})

# Check if the document was found
if result:
    # Get and print the food_preference
    food_preference = result.get("food_preference")
    print("Food Preference:", food_preference)
else:
    print("User 'Soumi' not found in the database.")
