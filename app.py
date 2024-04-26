from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import LabelEncoder
import pytesseract
from PIL import Image
import io
from bs4 import BeautifulSoup
from flask import jsonify


app = Flask(__name__)

# Function to perform OCR on an image
def perform_ocr(image):
    # Open the image using PIL
    img = Image.open(io.BytesIO(image.read()))
    # Perform OCR on the opened image
    text = pytesseract.image_to_string(img)
    return text

# Load the saved models
with open('model_natural.pkl', 'rb') as f:
    model_natural = pickle.load(f)

with open('model_processed.pkl', 'rb') as f:
    model_processed = pickle.load(f)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Function to predict classification of ingredients
def predict_ingredient_classification(ingredient):
    try:
        # Encode the input
        ingredient_encoded = label_encoder.transform([ingredient]).reshape(1, -1)

        # Predict for natural/artificial
        prediction_natural = model_natural.predict(ingredient_encoded)

        # Predict for processed/unprocessed
        prediction_processed = model_processed.predict(ingredient_encoded)

        # Decode predictions
        prediction_natural_decoded = "Natural" if prediction_natural == 0 else "Artificial"
        prediction_processed_decoded = "Unprocessed" if prediction_processed == 0 else "Processed"

        return prediction_natural_decoded, prediction_processed_decoded
    except ValueError as e:
        print(f"Ignoring unseen ingredient: {ingredient}")
        return None, None

# Function to scrape ingredient information using BeautifulSoup
def scrape_ingredient_info(search_term):
    # Load the HTML content from the file
    with open("sample.html", "r") as f:
        html_content = f.read()

    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all <tr> elements
    all_tr = soup.find_all('tr')

    # Flag to check if search term is found
    search_found = False

    # Array to store obtained values
    obtained_values = []

    # Loop through each <tr> element
    for tr in all_tr:
        # Find all <td> elements within the <tr>
        all_td = tr.find_all('td')
        
        # Check if the search term is in any of the <td> elements
        exact_match = False
        for td in all_td:
            if search_term.lower() == td.get_text(strip=True).lower():
                exact_match = True
                break
        
        # If the exact match is found in any <td> element
        if exact_match:
            # Set the flag to True
            search_found = True
            
            # Extract the text from the first <td> element containing the search term
            search_name = search_term
            print("Search Term:", search_term)
            
            # Loop through each <td> element in the same <tr>
            for td in all_td:
                # Get all siblings of the found <td> element
                siblings = td.find_next_siblings()
                for sibling in siblings:
                    # Append the text of the sibling to the array
                    obtained_values.append(sibling.get_text(strip=True))
            
            # Break the loop after finding the search term
            break

    # If search term not found
    if not search_found:
        obtained_values = None

    return obtained_values

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the list of ingredients from the form submission
        ingredients = request.form.get('ingredients')
        
        # Split the input by commas to extract individual ingredients
        ingredient_list = [ingredient.strip() for ingredient in ingredients.split(',')]

        # Process each ingredient separately and aggregate the results
        predictions = []
        for ingredient in ingredient_list:
            prediction = predict_ingredient_classification(ingredient)
            predictions.append(prediction)

        # Return the predictions as JSON
        return jsonify(predictions)


# Define route for handling image upload and ingredient classification
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    # Perform OCR on the uploaded image
    extracted_text = perform_ocr(file)
    
    # Split the extracted text into lines to extract ingredients
    lines = extracted_text.split('\n')
    predicted_ingredients = []
    not_predicted_ingredients = []  # Array to store ingredients not predicted by the model

    # Predict classification for each ingredient and store in a list
    for line in lines:
        # Split the line by commas to handle multiple ingredients
        ingredients = line.split(',')
        for ingredient in ingredients:
            # Strip whitespace and capitalize the first letter of the ingredient
            ingredient = ingredient.strip().capitalize()
            # Skip empty ingredients
            if ingredient:
                # Predict classification for the ingredient
                prediction_natural, prediction_processed = predict_ingredient_classification(ingredient)
                if prediction_natural is not None and prediction_processed is not None:
                    # Append the prediction to the list
                    predicted_ingredients.append((ingredient, prediction_natural, prediction_processed))
                else:
                    # If the ingredient was not predicted, log it
                    print(f"Ingredient not predicted: {ingredient}")
                    not_predicted_ingredients.append(ingredient)

    # Scrape information for unidentified ingredients
    ingredients_info = []
    for ingredient in not_predicted_ingredients:
        info = scrape_ingredient_info(ingredient)
        if info:
            ingredients_info.append({'ingredient': ingredient, 'info': info})

    # Render the result template with the predicted ingredients and scraped information
    return render_template('result.html', predicted_ingredients=predicted_ingredients, ingredients_info=ingredients_info)

if __name__ == '__main__':
    app.run(debug=True)
