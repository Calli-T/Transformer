import os


class VerseIterator:
    def __init__(self, split):
        self.source_path = os.getcwd() + '\\' + "source"
        self.from_path = self.source_path + '\\' + "from"
        self.to_path = self.source_path + '\\' + "to"

        self.book_list = os.listdir(self.from_path)
        self.f_from = open(self.from_path + '\\' + self.book_list[0])
        self.f_to = open(self.to_path + '\\' + self.book_list[0])
        self.book_count = 1

        self.split = split
        self.line_count = 0

    def get_one_verse(self):
        verse1 = self.f_from.readline()
        verse2 = self.f_to.readline()

        if verse1 == '':
            self.f_from.close()
            self.f_to.close()

            if self.book_count < 66:
                self.f_from = open(self.from_path + '\\' + self.book_list[self.book_count])
                self.f_to = open(self.to_path + '\\' + self.book_list[self.book_count])

                self.book_count += 1

                verse1 = self.f_from.readline()
                verse2 = self.f_to.readline()

            else:
                return []

        return [verse1.rstrip(), verse2.rstrip()]

    def __iter__(self):
        return self

    def __next__(self):
        self.line_count += 1

        if self.split == "train":
            if self.line_count > 28944:
                raise StopIteration

            if self.line_count % 29 == 27:
                _ = self.get_one_verse()
                _ = self.get_one_verse()

            return self.get_one_verse()

        elif self.split == "valid":
            if self.line_count > 1072:
                raise StopIteration

            if self.line_count == 1:
                for i in range(27):
                    _ = self.get_one_verse()
            else:
                for i in range(28):
                    _ = self.get_one_verse()

            return self.get_one_verse()

        elif self.split == "test":
            if self.line_count > 1072:
                raise StopIteration

            for i in range(28):
                _ = self.get_one_verse()
            return self.get_one_verse()

    def __len__(self):
        if self.split == "train":
            return 1072 * 27
        else:
            return 1072

    def __getitem__(self, item):
        return self.__next__()


# 1