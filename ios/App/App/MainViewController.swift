import Capacitor

/// Capacitor builds its plugin list from `packageClassList`, which the CLI fills in from
/// installed npm packages. A plugin that lives in the app itself is in no package, so
/// nothing registers it and every call answers "not implemented on ios".
class MainViewController: CAPBridgeViewController {
    override func capacitorDidLoad() {
        bridge?.registerPluginInstance(MagnetometerPlugin())
    }
}
