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
    

def transform_index_to_letter(batch_indices):
    '''
    Transforms numerical index input to string output by converting each index 
    to its corresponding letter from LETTER_LIST

    Args:
        batch_indices: List of indices from LETTER_LIST with the shape of (N, )
    
    Return:
        transcripts: List of converted string transcripts. This would be a list with a length of N
    '''
    transcripts = []
    # TODO
    # import pdb; pdb.set_trace()
    # for batch, len in zip(batch_indices, batch_len):
    #     string_output = ''
    #     for idx in range(len-1):
    #         string_output += index2letter[batch[idx]]
    #     transcripts.append(string_output)
    # return transcripts
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
        