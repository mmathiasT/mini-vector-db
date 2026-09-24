#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <string>

const size_t BUFFER_SIZE = 1024;
constexpr int DEFAULT_PORT = 8080;   // Must match the port run_server() is started with.

int main(int argc, char* argv[]) {
    std::string arg;

    if (argc < 2) {
        std::cout << "Usage: ./client QUERY|INSERT <values...>\n";
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        arg += argv[i];
        arg += " ";
    }
    arg += '\n';

    int server_file_descriptor = socket(AF_INET, SOCK_STREAM, 0);
    if (server_file_descriptor == -1) {
        std::cout << "socket() failed" << std::endl;
        return 1;
    }

    sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;                   // Use IPv4 addressing.
    address.sin_port = htons(DEFAULT_PORT);         // Convert the port number to network byte order.

    // Convert the textual IP address to the binary format sockaddr_in expects.
    inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);

    if (connect(server_file_descriptor, (sockaddr*)&address, sizeof(address)) == -1) {
        std::cout << "connect() failed - is the server running?" << std::endl;
        return 1;
    }
    write(server_file_descriptor, arg.c_str(), arg.size());

    char buffer[BUFFER_SIZE];
    int bytes_read = read(server_file_descriptor, buffer, sizeof(buffer));
    std::string response(buffer, bytes_read);

    std::cout << response;

    close(server_file_descriptor);
    return 0;
}
