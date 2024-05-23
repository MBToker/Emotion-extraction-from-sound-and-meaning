
def prediction_percentages(encoder, predictions, name_list):
    possible_encodings = encoder.categories_[0] # First element is a list of possible emotions
    softmax_probabilities = tf.nn.softmax(predictions) 

    for i, prediction in enumerate(softmax_probabilities): # Loops through every element and also keeps the index
        decoded_emotion = possible_encodings[np.argmax(prediction)]
        print(f"\nPrediction of file: {i}: {decoded_emotion}")
        
        # Creating a figure and axis
        fig, ax = plt.subplots()

        # Creating the bar chart
        bars = ax.bar(possible_encodings, prediction, color='skyblue')
        ax.set_ylabel('Probability')
        ax.set_title(f'Prediction {i+1}: {decoded_emotion}, file name: {name_list}')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Adding percentage labels on top of each bar
        for bar, percentage in zip(bars, prediction):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, '%.2f%%' % (percentage * 100),
                    ha='center', va='bottom')
            
        saved_path = 'graphs/percentages.png'
        plt.savefig(saved_path) 
        #plt.show()
        return saved_path