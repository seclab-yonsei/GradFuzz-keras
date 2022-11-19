def basicMetadataFunction(self, metadata_):
    metadata = []
    rooms = dict()
    for i in range(len(metadata_)):
        rooms[i] = dict()
        for j in range(len(metadata_[i])):
            rooms[i][j] = metadata_[i][j]

    for j in range(len(metadata_[0])):
        item = []
        for i in range(len(metadata_)):
            item.append(metadata_[i][j])
        metadata.append(item)

    return metadata

class NRFMetadata():

    def __init__(self, metadataFunction=basicMetadataFunction):
        self.metadataFunction=metadataFunction

    def checkMetadata(self, metadata_):
        return self.metadataFunction(self, metadata_)

