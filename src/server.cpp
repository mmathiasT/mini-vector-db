#include "server.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

constexpr size_t BUFFER_SIZE = 1024;

std::string read_line(int client_fd) {
    std::string request;
    char buffer[BUFFER_SIZE];

    while (true) {
        int bytes_read = read(client_fd, buffer, sizeof(buffer));

        if (bytes_read <= 0) {
            break;
        }

        request.append(buffer, bytes_read);

        size_t newline_pos = request.find('\n');
        if (newline_pos != std::string::npos) {
            // There is a newline sign at the newline_pos position.
            request = request.substr(0, newline_pos);
            break;
        }
    }

    return request;
}

void run_server(IVFIndex& ivf_index, int port) {
    int server_file_descriptor = socket(AF_INET, SOCK_STREAM, 0);
    if (server_file_descriptor == -1) {
        std::cout << "socket() failed" << std::endl;
        return;
    }

    sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;           // Use IPv4 addressing.
    address.sin_addr.s_addr = INADDR_ANY;   // Listen on all available network interfaces, not just one.
    address.sin_port = htons(port);         // Convert the port number to network byte order.

    // Associate the socket with the address/port configured above.
    if (bind(server_file_descriptor, (sockaddr*)&address, sizeof(address)) == -1) {
        std::cout << "bind() failed - port might already be in use" << std::endl;
        return;
    }

    if (listen(server_file_descriptor, 5) == -1) {
        std::cout << "listen() failed" << std::endl;
        return;
    }

    std::cout << "Server listening on port " << port << std::endl;

    while (true) {
        int client_file_descriptor = accept(server_file_descriptor, nullptr, nullptr);
        std::cout << "Client connected" << std::endl;

        std::string request = read_line(client_file_descriptor);

        std::istringstream iss(request);
        std::string command;
        iss >> command;

        if (command == "QUERY") {
            std::cout << "QUERY" << std::endl;
            size_t dim = ivf_index.data[0].data.size();

            Vec query;

            for (size_t i = 0; i < dim; i++) {
                float x;
                iss >> x;
                query.data.push_back(x);
            }

            int k, nprobe;
            iss >> k >> nprobe;

            std::vector<int> result = ivf_search(ivf_index, query, k, nprobe);
            
            std::string response;
            for (int x : result) {
                response += std::to_string(x) + " ";
            }
            response += "\n";

            write(client_file_descriptor, response.c_str(), response.size());
        }
        else if (command == "INSERT") {
            std::cout << "INSERT" << std::endl;
            size_t dim = ivf_index.data[0].data.size();

            Vec new_vector;

            new_vector.id = ivf_index.data.size();

            for (size_t i = 0; i < dim; i++) {
                float x;
                iss >> x;
                new_vector.data.push_back(x);
            }
            ivf_insert(ivf_index, new_vector);

            std::string response = "OK\n";
            write(client_file_descriptor, response.c_str(), response.size());
        }

        close(client_file_descriptor);
    }
}