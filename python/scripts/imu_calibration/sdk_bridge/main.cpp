// Read-only telemetry consumer. No motor, button or bootloader commands.
#include <steerable_instrument_driver/steerable_instrument_driver.hpp>
#include <steerable_instrument_driver/logging.hpp>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <thread>
#include <variant>

namespace sid = steerable_instrument_driver;
void emit(const char* source, std::uint32_t timestamp, const sid::Quaternion& q, bool valid) {
    const double n = std::sqrt(double(q.w)*q.w+double(q.x)*q.x+double(q.y)*q.y+double(q.z)*q.z);
    if (!std::isfinite(n) || n < 1e-6) return;
    std::cout << std::setprecision(10) << "{\"type\":\"attitude\",\"source\":\"" << source
              << "\",\"device_timestamp_ms\":" << timestamp << ",\"valid\":" << (valid?"true":"false")
              << ",\"quaternion_wxyz\":[" << q.w/n << ',' << q.x/n << ',' << q.y/n << ',' << q.z/n << "]}" << std::endl;
}
int main() {
    sid::set_log_level(sid::LogLevel::warning);
    for (;;) {
        sid::Device device;
        if (!device.open_first()) {
            std::cout << "{\"type\":\"connection\",\"connected\":false}" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(2)); continue;
        }
        if (!device.replace_subscriptions({{sid::DataStream::icm42670_pose,50},
                                          {sid::DataStream::jy931_attitude,50},
                                          {sid::DataStream::jy931_status,2}})) {
            std::cerr << "Subscription failed: " << device.last_error() << std::endl;
            continue;
        }
        std::cout << "{\"type\":\"connection\",\"connected\":true}" << std::endl;
        auto heartbeat = std::chrono::steady_clock::now();
        while (device.is_open()) {
            if (auto event=device.wait_event(std::chrono::milliseconds(250))) {
                if (auto p=std::get_if<sid::Icm42670Pose>(&*event)) emit("icm42670",p->device_timestamp_ms,p->quaternion,p->valid);
                if (auto p=std::get_if<sid::Jy931Attitude>(&*event)) emit("jy931",p->device_timestamp_ms,p->quaternion,true);
                if (auto p=std::get_if<sid::Jy931Status>(&*event))
                    std::cout << "{\"type\":\"jy931_status\",\"device_timestamp_ms\":" << p->device_timestamp_ms
                              << ",\"valid_flags\":" << unsigned(p->valid_flags) << ",\"checksum_error_count\":" << p->checksum_error_count << "}" << std::endl;
            }
            if (std::chrono::steady_clock::now()-heartbeat > std::chrono::seconds(1)) {
                heartbeat=std::chrono::steady_clock::now();
                std::cout << "{\"type\":\"connection\",\"connected\":true}" << std::endl;
            }
        }
    }
}
