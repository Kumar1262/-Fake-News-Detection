import pickle


def detecting_fake_news(var):
    try:
        load_model = pickle.load(open('final_model.sav','rb'))
    except FileNotFoundError:
        print("Model file not found. Make sure 'final_model.sav' exists in the current directory.")
        return

    prediction = load_model.predict([var])
    prob = load_model.predict_proba([var])

    return prediction[0], prob[0][1]


if __name__ == '__main__':
    var = input("Please enter the news text you want to verify: ").strip()
    if not var:
        print("No input provided. Exiting...")
    else:
        print("You entered:", var)
        result, probability = detecting_fake_news(var)
        print("The given statement is", result)
        print("The truth probability score is", probability)
