import Levenshtein as lev
import matplotlib.pyplot as plt
import seaborn as sns

def create_dictionaries(letter_list):
    '''
    Create dictionaries for letter2index and index2letter transformations
    based on LETTER_LIST

    Args:
        letter_list: LETTER_LIST

    Return:
        letter2index: Dictionary mapping from letters to indices
        index2letter: Dictionary mapping from indices to letters
    '''
    letter2index = dict()
    index2letter = dict()
    # TODO
    for i, l in enumerate(letter_list):
        letter2index[l] = i
        index2letter[i] = l
    return letter2index, index2letter

def transform_index_to_letter(batch_indices, index2letter):
    '''
    Transforms numerical index input to string output by converting each index 
    to its corresponding letter from LETTER_LIST

    Args:
        batch_indices: List of indices from LETTER_LIST with the shape of (N, )
    
    Return:
        transcripts: List of converted string transcripts. This would be a list with a length of N
    '''
    transcripts = []
    for batch in batch_indices:
        string_output = ''
        for idx in batch:
            letter = index2letter[idx]
            if letter == '<eos>':
                break
            else:
                string_output += letter
        transcripts.append(string_output)
    return transcripts

def calc_edit_distance(batch_text_1, batch_text_2):
    res = 0.0
    # import pdb; pdb.set_trace()
    for i, j in zip(batch_text_1, batch_text_2):
        distance = lev.distance(i, j)
        res += distance
    return res 

def plot_attention(attention, outpath):
    # utility function for debugging
    plt.clf()
    fig = sns.heatmap(attention, cmap='GnBu')
    plt.show()
    plt.savefig(outpath)

def to_csv(outpath, pred):
    with open(outpath, "w+") as f:
        f.write("id,predictions\n")
        for i in range(len(pred)):
            f.write("{},{}\n".format(i, pred[i]))

# The labels of the dataset contain letters in LETTER_LIST.
# You should use this to convert the letters to the corresponding indices
# and train your model with numerical labels.
LETTER_LIST = ['<sos>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
         'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', "'", ' ', '<eos>']

# Create the letter2index and index2letter dictionary
letter2index, index2letter = create_dictionaries(LETTER_LIST)
