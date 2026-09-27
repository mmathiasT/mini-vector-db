import socket
import sys

from embed_corpus import get_sentence_embedding

HOST = "127.0.0.1"
PORT = 8080

WORDS_PATH = "unique_words.txt"
with open(WORDS_PATH, "r") as f:
    unique_words = [line.strip() for line in f]

def query(text, k=5, nprobe=20):
    embedding = get_sentence_embedding(text)

    message = "QUERY "
    for x in embedding:
        message += str(x)
        message += " "
    message += str(k)
    message += " "
    message += str(nprobe)
    message += "\n"

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    # Create socket.
    sock.connect((HOST, PORT))                                  
    sock.sendall(message.encode())                              # Write the message.
    response = sock.recv(4096).decode()                         # Read response data.
    sock.close()

    ids = []
    for x in response.split():
        ids.append(int(x))

    words_found = []
    for i in ids:
        if 0 <= i < len(unique_words):
            words_found.append(unique_words[i])

    return words_found

if __name__ == "__main__":
    query_text = " ".join(sys.argv[1:])
    results = query(query_text)

    print(f"Query: {query_text!r}\n")
    for i, word in enumerate(results, 1):
        print(f"{i}. {word}")