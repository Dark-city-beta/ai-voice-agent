using System;
using System.Drawing;
using System.IO;
using System.Net;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Windows.Forms;

namespace GovorilkaTray
{
    internal static class Program
    {
        [STAThread]
        private static void Main(string[] args)
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm(LoadServer(args)));
        }

        private static string LoadServer(string[] args)
        {
            if (args != null && args.Length > 0 && !string.IsNullOrWhiteSpace(args[0]))
                return args[0].TrimEnd('/');

            string cfg = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "config.json");
            if (File.Exists(cfg))
            {
                string text = File.ReadAllText(cfg, Encoding.UTF8);
                Match m = Regex.Match(text, "\"server\"\\s*:\\s*\"([^\"]+)\"");
                if (m.Success)
                    return m.Groups[1].Value.TrimEnd('/');
            }
            return "http://192.168.31.200:8766";
        }
    }

    internal sealed class MainForm : Form
    {
        private const int WM_HOTKEY = 0x0312;
        private const int HOTKEY_TOGGLE = 1;
        private const int HOTKEY_MIC = 2;
        private const int HOTKEY_SPEAKER = 3;
        private const uint MOD_ALT = 0x0001;
        private const uint MOD_CONTROL = 0x0002;

        [DllImport("user32.dll")]
        private static extern bool RegisterHotKey(IntPtr hWnd, int id, uint fsModifiers, uint vk);

        [DllImport("user32.dll")]
        private static extern bool UnregisterHotKey(IntPtr hWnd, int id);

        private readonly string server;
        private readonly NotifyIcon tray;
        private readonly Timer timer;
        private readonly Label statusLabel;
        private readonly Button toggleButton;
        private readonly Button micButton;
        private readonly Button speakerButton;
        private readonly Button refreshButton;
        private readonly Button hideButton;
        private readonly TextBox logBox;
        private bool allowClose;

        public MainForm(string serverUrl)
        {
            server = serverUrl;

            Text = "Говорилка 0.3";
            StartPosition = FormStartPosition.CenterScreen;
            Size = new Size(460, 330);
            MinimumSize = new Size(430, 300);
            MaximizeBox = false;
            Font = new Font("Segoe UI", 9F);

            Label title = new Label();
            title.Text = "Говорилка 0.3";
            title.Font = new Font("Segoe UI", 14F, FontStyle.Bold);
            title.Location = new Point(16, 12);
            title.Size = new Size(240, 30);
            Controls.Add(title);

            Label serverLabel = new Label();
            serverLabel.Text = "Сервер: " + server;
            serverLabel.Location = new Point(18, 48);
            serverLabel.Size = new Size(410, 22);
            Controls.Add(serverLabel);

            statusLabel = new Label();
            statusLabel.Text = "Статус: проверяю...";
            statusLabel.Location = new Point(18, 76);
            statusLabel.Size = new Size(410, 44);
            Controls.Add(statusLabel);

            toggleButton = MakeButton("Включить / выключить", 18, 128, 200, 36);
            toggleButton.Click += delegate { ToggleVoice(); };
            Controls.Add(toggleButton);

            micButton = MakeButton("Микрофон mute", 230, 128, 190, 36);
            micButton.Click += delegate { ToggleMic(); };
            Controls.Add(micButton);

            speakerButton = MakeButton("Колонки mute", 18, 172, 200, 36);
            speakerButton.Click += delegate { ToggleSpeaker(); };
            Controls.Add(speakerButton);

            refreshButton = MakeButton("Обновить статус", 230, 172, 190, 36);
            refreshButton.Click += delegate { RefreshStatus(true); };
            Controls.Add(refreshButton);

            hideButton = MakeButton("Свернуть в трей", 18, 216, 200, 36);
            hideButton.Click += delegate { HideToTray(); };
            Controls.Add(hideButton);

            Button exitButton = MakeButton("Выход", 230, 216, 190, 36);
            exitButton.Click += delegate { allowClose = true; Close(); };
            Controls.Add(exitButton);

            logBox = new TextBox();
            logBox.Location = new Point(18, 260);
            logBox.Size = new Size(402, 24);
            logBox.ReadOnly = true;
            logBox.Text = "Ctrl+Alt+G старт/стоп, Ctrl+Alt+M микрофон, Ctrl+Alt+S колонки";
            Controls.Add(logBox);

            tray = new NotifyIcon();
            tray.Icon = SystemIcons.Application;
            tray.Text = "Говорилка";
            tray.Visible = true;
            tray.ContextMenuStrip = BuildMenu();
            tray.DoubleClick += delegate { ShowWindow(); };

            timer = new Timer();
            timer.Interval = 5000;
            timer.Tick += delegate { RefreshStatus(false); };
            timer.Start();
        }

        private static Button MakeButton(string text, int x, int y, int w, int h)
        {
            Button b = new Button();
            b.Text = text;
            b.Location = new Point(x, y);
            b.Size = new Size(w, h);
            return b;
        }

        protected override void OnLoad(EventArgs e)
        {
            base.OnLoad(e);
            RegisterHotKey(Handle, HOTKEY_TOGGLE, MOD_ALT | MOD_CONTROL, (uint)Keys.G);
            RegisterHotKey(Handle, HOTKEY_MIC, MOD_ALT | MOD_CONTROL, (uint)Keys.M);
            RegisterHotKey(Handle, HOTKEY_SPEAKER, MOD_ALT | MOD_CONTROL, (uint)Keys.S);
            RefreshStatus(false);
        }

        protected override void OnResize(EventArgs e)
        {
            base.OnResize(e);
            if (WindowState == FormWindowState.Minimized)
                HideToTray();
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            if (!allowClose && e.CloseReason == CloseReason.UserClosing)
            {
                e.Cancel = true;
                HideToTray();
                return;
            }
            base.OnFormClosing(e);
        }

        protected override void WndProc(ref Message m)
        {
            if (m.Msg == WM_HOTKEY)
            {
                int id = m.WParam.ToInt32();
                if (id == HOTKEY_TOGGLE) ToggleVoice();
                else if (id == HOTKEY_MIC) ToggleMic();
                else if (id == HOTKEY_SPEAKER) ToggleSpeaker();
            }
            base.WndProc(ref m);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                UnregisterHotKey(Handle, HOTKEY_TOGGLE);
                UnregisterHotKey(Handle, HOTKEY_MIC);
                UnregisterHotKey(Handle, HOTKEY_SPEAKER);
                timer.Dispose();
                tray.Visible = false;
                tray.Dispose();
            }
            base.Dispose(disposing);
        }

        private ContextMenuStrip BuildMenu()
        {
            ContextMenuStrip menu = new ContextMenuStrip();

            ToolStripMenuItem open = new ToolStripMenuItem("Открыть окно");
            open.Click += delegate { ShowWindow(); };
            menu.Items.Add(open);

            ToolStripMenuItem toggle = new ToolStripMenuItem("Включить / выключить    Ctrl+Alt+G");
            toggle.Click += delegate { ToggleVoice(); };
            menu.Items.Add(toggle);

            ToolStripMenuItem mic = new ToolStripMenuItem("Микрофон mute/unmute    Ctrl+Alt+M");
            mic.Click += delegate { ToggleMic(); };
            menu.Items.Add(mic);

            ToolStripMenuItem speaker = new ToolStripMenuItem("Колонки mute/unmute    Ctrl+Alt+S");
            speaker.Click += delegate { ToggleSpeaker(); };
            menu.Items.Add(speaker);

            menu.Items.Add(new ToolStripSeparator());

            ToolStripMenuItem exit = new ToolStripMenuItem("Выход");
            exit.Click += delegate { allowClose = true; Close(); };
            menu.Items.Add(exit);

            return menu;
        }

        private string Request(string method, string path)
        {
            HttpWebRequest req = (HttpWebRequest)WebRequest.Create(server + path);
            req.Method = method;
            req.Timeout = 8000;
            req.ReadWriteTimeout = 8000;
            if (method == "POST")
            {
                byte[] body = Encoding.UTF8.GetBytes("");
                req.ContentLength = body.Length;
                using (Stream stream = req.GetRequestStream())
                    stream.Write(body, 0, body.Length);
            }

            using (HttpWebResponse resp = (HttpWebResponse)req.GetResponse())
            using (StreamReader reader = new StreamReader(resp.GetResponseStream(), Encoding.UTF8))
                return reader.ReadToEnd();
        }

        private VoiceStatus FetchStatus()
        {
            return VoiceStatus.Parse(Request("GET", "/status"));
        }

        private void ToggleVoice()
        {
            try
            {
                VoiceStatus st = FetchStatus();
                Request("POST", st.Running ? "/voice/stop" : "/voice/start");
                RefreshStatus(true);
            }
            catch (Exception ex) { ShowError(ex); }
        }

        private void ToggleMic()
        {
            try
            {
                VoiceStatus st = FetchStatus();
                Request("POST", st.MicMuted ? "/mic/unmute" : "/mic/mute");
                RefreshStatus(true);
            }
            catch (Exception ex) { ShowError(ex); }
        }

        private void ToggleSpeaker()
        {
            try
            {
                VoiceStatus st = FetchStatus();
                Request("POST", st.SpeakerMuted ? "/speaker/unmute" : "/speaker/mute");
                RefreshStatus(true);
            }
            catch (Exception ex) { ShowError(ex); }
        }

        private void RefreshStatus(bool showErrors)
        {
            try
            {
                VoiceStatus st = FetchStatus();
                string run = st.Running ? "включена" : "выключена";
                statusLabel.Text =
                    "Статус: " + run + " / " + st.State + Environment.NewLine +
                    "Микрофон: " + (st.MicMuted ? "mute" : "on") +
                    "    Колонки: " + (st.SpeakerMuted ? "mute" : "on");
                logBox.Text = DateTime.Now.ToString("HH:mm:ss") + " статус обновлён";
                SetTrayText("Говорилка: " + (st.Running ? "on" : "off") + " / " + st.State);
            }
            catch (Exception ex)
            {
                statusLabel.Text = "Статус: сервер недоступен";
                logBox.Text = ex.Message;
                SetTrayText("Говорилка: сервер недоступен");
                if (showErrors)
                    ShowError(ex);
            }
        }

        private void HideToTray()
        {
            Hide();
            WindowState = FormWindowState.Minimized;
            tray.Visible = true;
        }

        private void ShowWindow()
        {
            Show();
            WindowState = FormWindowState.Normal;
            Activate();
        }

        private void SetTrayText(string text)
        {
            if (text.Length > 63)
                text = text.Substring(0, 63);
            tray.Text = text;
        }

        private void ShowError(Exception ex)
        {
            MessageBox.Show(
                "Не удалось связаться с Говорилкой:" + Environment.NewLine + ex.Message,
                "Говорилка",
                MessageBoxButtons.OK,
                MessageBoxIcon.Warning);
        }
    }

    internal sealed class VoiceStatus
    {
        public bool Running;
        public bool MicMuted;
        public bool SpeakerMuted;
        public string State = "unknown";

        public static VoiceStatus Parse(string json)
        {
            VoiceStatus st = new VoiceStatus();
            st.Running = BoolField(json, "running");
            st.MicMuted = BoolField(json, "mic_muted");
            st.SpeakerMuted = BoolField(json, "speaker_muted");

            Match nested = Regex.Match(json, "\"state\"\\s*:\\s*\\{[^}]*\"state\"\\s*:\\s*\"([^\"]+)\"");
            if (nested.Success)
                st.State = nested.Groups[1].Value;
            else
            {
                Match flat = Regex.Match(json, "\"state\"\\s*:\\s*\"([^\"]+)\"");
                if (flat.Success)
                    st.State = flat.Groups[1].Value;
            }
            return st;
        }

        private static bool BoolField(string json, string name)
        {
            return Regex.IsMatch(json, "\"" + Regex.Escape(name) + "\"\\s*:\\s*true");
        }
    }
}
