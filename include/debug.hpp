#include <iostream>
#include <string>

void print_variables() {
}


template<typename T, typename... Args>
void print_variables(const std::string& name, const T& value, const Args&... args) {
    std::cout << "Varible Name: " << name << std::endl; 
    std::cout << "Varible Value:\n" <<  value << std::endl;
    print_variables(args...);
}