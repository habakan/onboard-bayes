import Capacitor
import CoreMotion

/// Raw magnetometer, which a WebView cannot reach on iOS: `webkitCompassHeading` is a
/// heading the OS has already fused, and the Generic Sensor API is not implemented.
///
/// `startMagnetometerUpdates` is the uncalibrated field, in microteslas. The other
/// route, `deviceMotion.magneticField`, has already had the hard-iron offset removed —
/// which is the very thing the calibration model is trying to estimate.
@objc(MagnetometerPlugin)
public class MagnetometerPlugin: CAPPlugin, CAPBridgedPlugin {
    public let identifier = "MagnetometerPlugin"
    public let jsName = "Magnetometer"
    public let pluginMethods: [CAPPluginMethod] = [
        CAPPluginMethod(name: "isAvailable", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "start", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "stop", returnType: CAPPluginReturnPromise),
    ]

    private let motion = CMMotionManager()

    @objc func isAvailable(_ call: CAPPluginCall) {
        call.resolve(["available": motion.isMagnetometerAvailable])
    }

    @objc func start(_ call: CAPPluginCall) {
        guard motion.isMagnetometerAvailable else {
            call.reject("magnetometer unavailable on this device")
            return
        }
        if motion.isMagnetometerActive {
            call.resolve()
            return
        }
        let hz = call.getDouble("hz") ?? 50.0
        motion.magnetometerUpdateInterval = 1.0 / max(1.0, hz)
        motion.startMagnetometerUpdates(to: .main) { [weak self] data, error in
            if let error {
                self?.notifyListeners("error", data: ["message": error.localizedDescription])
                return
            }
            guard let d = data else { return }
            self?.notifyListeners(
                "reading",
                data: [
                    "x": d.magneticField.x,
                    "y": d.magneticField.y,
                    "z": d.magneticField.z,
                    "t": d.timestamp,
                ])
        }
        call.resolve()
    }

    @objc func stop(_ call: CAPPluginCall) {
        motion.stopMagnetometerUpdates()
        call.resolve()
    }
}
