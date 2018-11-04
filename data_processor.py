import string

def read_data(data_path,encoding='utf-8'):
    data = []
    with open(data_path,'r',encoding=encoding) as f:
        data = f.readlines()
    return data


class Data():
    '''
    This class is used to manipulate text data as data vectors
    '''
    def __init__(self,data_path,vocab_path):
        self.data = read_data(data_path)
        self.vocab_dict,self.total_word_frequency = self.read_vocab(vocab_path)

    def read_vocab(self,vocab_path):
        vocab_dict = {}
        data = []
        total_frequency = 0
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = f.readlines()
        for line in data:
            temp = line.split("\t")

            index = temp[1]
            frequency = int(temp[2].strip('\n'))
            total_frequency+=frequency

            vocab_dict[index] = frequency

        return vocab_dict,total_frequency


    def get_binary_representation(self):
        X_array = []
        Y_vector = []
        for line in self.data:
            current_input = []
            word_list = line.split(" ")
            last_e = word_list[-1].split("\t")
            word_list[-1] = last_e[0]
            cls = last_e[1].strip("\n")

            for word in self.vocab_dict:
                if word in word_list:
                    current_input.append(1)
                else:
                    current_input.append(0)

            X_array.append(current_input)
            Y_vector.append(int(cls))

        return X_array, Y_vector

    def get_frequency_representation(self):
        X_array = []
        Y_vector = []
        for line in self.data:
            current_input = []

            word_list = line.split(" ")
            last_e = word_list[-1].split("\t")
            word_list[-1] = last_e[0]
            cls = last_e[1].strip("\n")

            total_words = len(word_list)
            for word in self.vocab_dict:
                if word in word_list:
                    num_occ = Data.count_occurences(word,word_list)
                    frequency = num_occ/total_words
                    current_input.append(frequency)
                else:
                    current_input.append(0)

            X_array.append(current_input)
            Y_vector.append(int(cls))

        return X_array, Y_vector


    def save_vector_representation(self,saving_path,vtype="binary"):
        vtypes = ("binary","frequency")
        if vtype not in vtypes:
            print("Vector type can only be binary or frequency")
            return None

        if vtype == "binary":
            x_array, y_vector = self.get_binary_representation()
        else:
            x_array, y_vector = self.get_frequency_representation()

        Data.create_x_array_csv(x_array, saving_path)
        Data.create_y_vector_csv(y_vector,saving_path)

    @staticmethod
    def read_x_array(data_path):
        x_array = []
        data = []
        with open(data_path,"r") as f:
            data = f.readlines()

        for line in data:
            temp = line.split(",")[:-1]
            row = []
            for e in temp:
                if e:
                    row.append(float(e))
            x_array.append(row)
        return x_array

    @staticmethod
    def read_y_array(data_path):
        y_array = []
        data = []
        with open(data_path,"r") as f:
            data = f.readlines()
        results = data[0].split(",")[:-1]
        for e in results:
            if e:
                y_array.append(float(e))
        return y_array

    @staticmethod
    def create_x_array_csv(x_array,saving_path):
        output = ""
        for line in x_array:
            for e in line:
                output+= str(e) + ","
            output+="\n"
        with open(saving_path+"-X.csv","w") as f:
            f.write(output)


    @staticmethod
    def create_y_vector_csv(y_vector,saving_path):
        output = ""
        for e in y_vector:
            output+= str(e) + ","
        with open(saving_path+"-Y.csv","w") as f:
            f.write(output)

    @staticmethod
    def count_occurences(val,l):
        count = 0
        for e in l:
            if e==val:
                count+=1
        return count

    @staticmethod
    def merge_arrays(arr_1,arr_2):
        new_arr = []
        for e in arr_1:
            new_arr.append(e)
        for e2 in arr_2:
            new_arr.append(e2)
        return new_arr



def pre_process_data(data):
    '''
    takes as input a list containing lines of data.
    Removes punctuation marks, changes every word to lowercase, and then
    returns a list containing a list of words and the last element is the class
    '''
    output = []
    to_remove = string.punctuation
    to_remove+="br"
    translator = str.maketrans("","",to_remove)
    for line in data:
        word_list = line.split(" ")
        final_word_list = []
        num_words = len(word_list)-1
        for i in range(0,num_words):
            word = word_list[i]
            word = word.translate(translator)
            word = word.lower()
            if word:
                final_word_list.append(word)

        last_words = word_list[-1].split('\t')
        final_word_list.append(last_words[0])
        final_word_list.append(last_words[1].strip('\n'))

        output.append(final_word_list)

    return output

def build_vocab(data):
    vocab_dict = {}
    for line in data:
        line_length = len(line)-1
        for i in range(0,line_length):
            if line[i] in vocab_dict:
                vocab_dict[line[i]] = vocab_dict[line[i]] + 1
            else:
                vocab_dict[line[i]] = 1

    sorted_list = []

    for word in sorted(vocab_dict, key=vocab_dict.get,reverse=True):
        sorted_list.append((word,vocab_dict[word]))
    return sorted_list


def save_vocab_file(vocab_list,saving_path,encoding='utf-8'):
    output = ""

    for i in range(0,len(vocab_list)):
        output = output + vocab_list[i][0] + "\t"
        output = output + str(i+1) + "\t"
        output = output + str(vocab_list[i][1]) + "\n"

    with open(saving_path,'w',encoding=encoding) as f:
        f.write(output)


def create_vocab_file(vocab_size,reading_path,saving_path):
    vocab_size = 10000

    data = read_data(reading_path)
    processed_data = pre_process_data(data)

    vocab_list = build_vocab(processed_data)[0:vocab_size]
    save_vocab_file(vocab_list, saving_path)


def read_vocab(vocab_path):
    vocab_dict = {}
    data = []
    with open(vocab_path,"r",encoding="utf-8") as f:
        data = f.readlines()
    for line in data:
        temp = line.split("\t")
        vocab_dict[temp[0]] = temp[1]
    return vocab_dict


def code_data(data,vocab):
    output = ""
    for line in data:
        num_words = len(line) - 2
        for i in range(0,num_words):
            if line[i] in vocab:
                output += vocab[line[i]] + " "

        output += line[-2] + "\t"
        output += line[-1] + "\n"

    return output

def create_coded_file(reading_path,saving_path,vocab_path):
    data = read_data(reading_path)
    processed_data = pre_process_data(data)
    vocab_dict = read_vocab(vocab_path)
    coded_data = code_data(processed_data,vocab_dict)
    with open(saving_path,"w",encoding="utf-8") as f:
        f.write(coded_data)


def prepare_data():
    datasets = ("IMDB","yelp")
    dataclasses= ("test","train","valid")

    for dataset in datasets:
        create_vocab_file(
            vocab_size=10000,
            reading_path='Data/Raw/{}-train.txt'.format(dataset),
            saving_path='Data/Processed/{}-vocab.txt'.format(dataset)
        )
        print("{}-vocab.txt file has been created!".format(dataset))

        for dataclass in dataclasses:
            create_coded_file(
                reading_path="Data/Raw/{}-{}.txt".format(dataset,dataclass),
                saving_path="Data/Processed/{}-{}.txt".format(dataset,dataclass),
                vocab_path="Data/Processed/{}-vocab.txt".format(dataset)
            )
            print("{}-{}.txt file has ben created!".format(dataset,dataclass))



def create_text_vectors():
    datasets = ("yelp","IMDB")
    dataclasses = ("test", "train", "valid")

    for dataset in datasets:
        for dataclass in dataclasses:

            data = Data(
                data_path="Data/Processed/{}-{}.txt".format(dataset,dataclass),
                vocab_path="Data/Processed/{}-vocab.txt".format(dataset)
            )
            data.save_vector_representation(
                saving_path="Data/BinaryBOW/{}-{}".format(dataset,dataclass),
                vtype="binary"
            )
            print("Data/BinaryBOW/{}-{}-X.csv file has been created!".format(dataset,dataclass))
            print("Data/BinaryBOW/{}-{}-Y.csv file has been created!".format(dataset,dataclass))

            data.save_vector_representation(
                saving_path="Data/FrequencyBOW/{}-{}".format(dataset,dataclass),
                vtype="frequency"
            )
            print("Data/FrequencyBOW/{}-{}-X.csv file has been created".format(dataset,dataclass))
            print("Data/FrequencyBOW/{}-{}-Y.csv file has been created".format(dataset,dataclass))


if __name__=="__main__":
    prepare_data()
    create_text_vectors()
