import UIKit
import AVFoundation
import os

final class M5ViewController: UIViewController {
    private let textView = UITextView()
    private let langPicker = UIPickerView()
    private let generateButton = UIButton(type: .system)
    private let stopButton = UIButton(type: .system)
    private let statusLabel = UILabel()
    private let perfLabel = UILabel()
    private let modelLabel = UILabel()
    private var engine: M5Engine?
    private var player: StreamingPlayer?
    private var cancelFlag = false
    private var workItem: DispatchWorkItem?
    private let languages = M5Language.allCases

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemBackground
        title = "M5 离线语音（iOS demo）"
        setupLayout()
        refreshModelStatus()
        try? AVAudioSession.sharedInstance().setCategory(.playback, options: [.mixWithOthers])
    }

    private func setupLayout() {
        let scroll = UIScrollView(); scroll.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(scroll)
        NSLayoutConstraint.activate([
            scroll.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
            scroll.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            scroll.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            scroll.trailingAnchor.constraint(equalTo: view.trailingAnchor),
        ])
        let root = UIStackView(); root.axis = .vertical; root.spacing = 12; root.translatesAutoresizingMaskIntoConstraints = false
        scroll.addSubview(root)
        NSLayoutConstraint.activate([
            root.topAnchor.constraint(equalTo: scroll.contentLayoutGuide.topAnchor, constant: 16),
            root.bottomAnchor.constraint(equalTo: scroll.contentLayoutGuide.bottomAnchor, constant: 16),
            root.leadingAnchor.constraint(equalTo: scroll.contentLayoutGuide.leadingAnchor, constant: 16),
            root.trailingAnchor.constraint(equalTo: scroll.contentLayoutGuide.trailingAnchor, constant: -16),
            root.widthAnchor.constraint(equalTo: scroll.frameLayoutGuide.widthAnchor, constant: -32),
        ])

        modelLabel.numberOfLines = 3; modelLabel.font = .systemFont(ofSize: 13); modelLabel.textColor = .secondaryLabel
        root.addArrangedSubview(modelLabel)

        let langRow = UIStackView(); langRow.axis = .horizontal; langRow.spacing = 8
        let langTitle = UILabel(); langTitle.text = "语言"; langTitle.font = .preferredFont(forTextStyle: .headline)
        langPicker.dataSource = self; langPicker.delegate = self
        langRow.addArrangedSubview(langTitle); langRow.addArrangedSubview(langPicker)
        root.addArrangedSubview(langRow)

        textView.font = .systemFont(ofSize: 17); textView.layer.borderColor = UIColor.separator.cgColor
        textView.layer.borderWidth = 1; textView.layer.cornerRadius = 8; textView.text = "你好，这是在 iPhone 上运行的离线语音合成测试。"
        textView.heightAnchor.constraint(equalToConstant: 110).isActive = true
        root.addArrangedSubview(textView)

        generateButton.setTitle("生成语音（边生成边播）", for: .normal); generateButton.titleLabel?.font = .preferredFont(forTextStyle: .title3)
        generateButton.addTarget(self, action: #selector(generateTapped), for: .touchUpInside)
        root.addArrangedSubview(generateButton)
        stopButton.setTitle("停止", for: .normal); stopButton.addTarget(self, action: #selector(stopTapped), for: .touchUpInside)
        stopButton.isHidden = true
        root.addArrangedSubview(stopButton)

        statusLabel.numberOfLines = 2; statusLabel.text = "准备就绪"
        root.addArrangedSubview(statusLabel)

        perfLabel.numberOfLines = 8; perfLabel.font = .monospacedSystemFont(ofSize: 13, weight: .regular)
        perfLabel.text = "生成后显示：首块延迟 / 生成耗时 / RTF / 实时倍数 / 峰值内存 / CPU"
        perfLabel.layer.borderColor = UIColor.separator.cgColor; perfLabel.layer.borderWidth = 1; perfLabel.layer.cornerRadius = 8
        root.addArrangedSubview(perfLabel)

        let hint = UILabel(); hint.numberOfLines = 0; hint.font = .systemFont(ofSize: 12); hint.textColor = .secondaryLabel
        hint.text = "模型导入：把与 Android 相同的 10 个模型文件（backbone.onnx 等，见 README）拷到本 App 的 Documents/m5-model/ 目录——iPhone 上用「文件」App，或访达里把设备当 U 盘拖入。iOS demo 暂不含文本正规化（TN），数字请写成汉字或英文单词。"
        root.addArrangedSubview(hint)
    }

    private func refreshModelStatus() {
        if let m = try? ModelDirectory(directory: ModelDirectory.defaultDirectory()) {
            modelLabel.text = "模型已导入 · \(m.voices.count) 个内置音色 · precision=\(m.precision) · 完全离线"
        } else {
            modelLabel.text = "未找到模型：请把模型目录拷到 Documents/m5-model/（10 个文件，见下方说明）"
        }
    }

    @objc private func generateTapped() {
        let text = textView.text ?? ""
        let lang = languages[langPicker.selectedRow(inComponent: 0)]
        cancelFlag = false
        generateButton.isEnabled = false; stopButton.isHidden = false
        perfLabel.text = "生成中…"
        let cpu0 = cpuSeconds(), rss0 = peakRSSMiB()
        let t0 = Date()
        let item = DispatchWorkItem { [weak self] in
            guard let self else { return }
            do {
                let model = try ModelDirectory(directory: ModelDirectory.defaultDirectory())
                if self.engine == nil {
                    #if canImport(OnnxRuntime)
                    let runner = try OnnxRuntimeRunner(model: model)
                    #else
                    let runner = UnavailableRunner()
                    #endif
                    self.engine = try M5Engine(model: model, runner: runner)
                }
                let voice = model.voices[0]
                try self.engine!.prepareVoice(voice)
                let player = try StreamingPlayer()
                self.player = player
                var cancel = self.cancelFlag
                let out = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("m5-last.wav")
                let synthesis = try self.engine!.synthesize(text: text, voice: voice, language: lang,
                                                            output: out, cancel: &cancel,
                                                            progress: { [weak self] in self?.status($0) },
                                                            sink: player)
                player.drainAndStop()
                let genMs = synthesis.elapsedSeconds * 1000, audioMs = synthesis.audioSeconds * 1000
                let rtf = audioMs > 0 ? genMs / audioMs : 0
                let realtime = genMs > 0 ? audioMs / genMs : 0
                let cpu = cpuSeconds() - cpu0, wall = Date().timeIntervalSince(t0)
                let cpuPct = wall > 0 ? cpu / wall * 100 : 0
                let peak = max(peakRSSMiB(), rss0)
                m5perf(String(format: "[M5_PERF] device=%@ precision=%@ lang=%@ chars=%d segments=%d audio_ms=%.0f gen_ms=%.0f wall_ms=%.0f rtf=%.3f realtime=%.2f first_chunk_ms=%.0f peak_rss_mib=%.0f cpu_pct=%.0f streaming=true",
                              Self.deviceModel(), self.engine!.precision, lang.rawValue, text.count, synthesis.segments,
                              audioMs, genMs, synthesis.wallSeconds * 1000, rtf, realtime, synthesis.firstChunkSeconds * 1000, peak, cpuPct))
                DispatchQueue.main.async {
                    self.perfLabel.text = String(format: "首块延迟  %.0f ms\n生成耗时  %.2f s · 音频 %.2f s\nRTF %.3f · 实时 %.2f×\n峰值内存  %.0f MiB · CPU 均 %.0f%%\n%d 段 · %d 帧 · 模型 %@",
                                                 synthesis.firstChunkSeconds * 1000, synthesis.elapsedSeconds, synthesis.audioSeconds,
                                                 rtf, realtime, peak, cpuPct, synthesis.segments, synthesis.frames, self.engine!.precision)
                    self.status("生成完成，已边生成边播放；WAV 已存 Documents/m5-last.wav")
                }
            } catch {
                self.player?.abort()
                let msg = (error as? LocalizedError)?.errorDescription ?? "\(error)"
                m5perf("[M5_FAIL] \(msg)")
                DispatchQueue.main.async { self.status("失败：\(msg)") }
            }
            DispatchQueue.main.async {
                self.generateButton.isEnabled = true; self.stopButton.isHidden = true
                self.refreshModelStatus()
            }
        }
        workItem = item
        DispatchQueue.global(qos: .userInitiated).async(execute: item)
    }

    @objc private func stopTapped() {
        cancelFlag = true
        status("正在停止…")
    }

    private func status(_ s: String) { DispatchQueue.main.async { self.statusLabel.text = s } }

    // MARK: 资源计量

    private func cpuSeconds() -> Double {
        var ru = rusage()
        getrusage(RUSAGE_SELF, &ru)
        return Double(ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) + Double(ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) / 1e6
    }
    private func peakRSSMiB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        } }
        return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1048576.0 : 0
    }
    static func deviceModel() -> String {
        var sys = utsname(); uname(&sys)
        return withUnsafeBytes(of: &sys.machine) { raw in
            String(decoding: raw.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
    }
}

extension M5ViewController: UIPickerViewDataSource, UIPickerViewDelegate {
    func numberOfComponents(in pickerView: UIPickerView) -> Int { 1 }
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int { languages.count }
    func pickerView(_ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int) -> String? { languages[row].display }
}

@main
final class AppDelegate: UIResponder, UIApplicationDelegate {
    var window: UIWindow?
    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil) -> Bool {
        window = UIWindow(frame: UIScreen.main.bounds)
        window?.rootViewController = UINavigationController(rootViewController: M5ViewController())
        window?.makeKeyAndVisible()
        return true
    }
}
