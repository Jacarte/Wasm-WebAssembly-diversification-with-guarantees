from happytransformer import  HappyTextToText, TTSettings
import numpy as np


def get_edit_path(D, tokens_original, tokens_corrected):
    """
    Returns a list of tuples of (original, corrected) words
    """

    i = len(tokens_original)
    j = len(tokens_corrected)

    edit_path = []

    while (i > 0 or j > 0):
        if (i > 0 and j > 0 and tokens_original[i - 1] == tokens_corrected[j - 1]):
            edit_path.append((tokens_original[i - 1], i- 1, tokens_corrected[j - 1]))
            i = i - 1
            j = j - 1
        elif (i > 0 and D[i][j] == D[i - 1][j] + 1):
            edit_path.append((tokens_original[i - 1], i -1,  ""))
            i = i - 1
        elif (j > 0 and D[i][j] == D[i][j - 1] + 1):
            edit_path.append(("", -1, tokens_corrected[j - 1]))
            j = j - 1
        else:
            edit_path.append((tokens_original[i - 1], i-1, tokens_corrected[j - 1]))
            i = i - 1
            j = j - 1

    return edit_path

def get_levensthein_edit_path(original, corrected):
    """
    Returns a list of tuples of (original, corrected) words
    """
    tokens_original = original.split()
    tokens_corrected = corrected.split()

    distances = np.zeros((len(tokens_original) + 1, len(tokens_corrected) + 1))

    a = 0
    b = 0
    c = 0

    for t1 in range(len(tokens_original) + 1):
        distances[t1][0] = t1

    for t2 in range(len(tokens_corrected) + 1):
        distances[0][t2] = t2

    for t1 in range(1, len(tokens_original) + 1):
        for t2 in range(1, len(tokens_corrected) + 1):
            if (tokens_original[t1-1] == tokens_corrected[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

            if (a <= b and a <= c):
                distances[t1][t2] = a + 1
            elif (b <= a and b <= c):
                distances[t1][t2] = b + 1
            else:
                distances[t1][t2] = c + 1

    D = distances[len(tokens_original)][len(tokens_corrected)]

    # Now get the edit path

    edit_path = get_edit_path(distances, tokens_original, tokens_corrected)
    
    return edit_path
    

if __name__ == '__main__':
    #model = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
    #beam_settings =  TTSettings(num_beams=5, min_length=1, max_length=100000)

    original_text = 'I I wants to codes.'
    #get_levensthein_edit_path("kelm", "helm")
    #output_text_1 = model.generate_text(f"grammar: {original_text}", args=beam_settings)

    #print(output_text_1.text)
    #get_levensthein_edit_path(original_text, output_text_1.text)
    get_levensthein_edit_path(original_text, "want to code.")
    