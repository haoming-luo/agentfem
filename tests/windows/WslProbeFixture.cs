// Deterministic executable for pipe/timeout tests; never invokes Windows WSL.
using System;
using System.IO;
using System.Text;
using System.Threading;

class WslProbeFixture
{
    public static int Main(string[] args)
    {
        Console.OutputEncoding = Encoding.Unicode;
        string mode = Environment.GetEnvironmentVariable("AGENTFEM_WSL_FIXTURE_MODE");
        if (!String.IsNullOrEmpty(mode))
        {
            string log = Environment.GetEnvironmentVariable("AGENTFEM_WSL_FIXTURE_LOG");
            if (!String.IsNullOrEmpty(log)) File.AppendAllText(log, String.Join(" ", args) + "\n");
            if (mode == "denied") return unchecked((int)0x80070005);
            if (mode == "timeout") { Thread.Sleep(10000); return 0; }
            if (args.Length == 1 && args[0] == "--version")
            {
                Console.WriteLine(mode == "inventory-timeout" ? "WSL version: 2.6.1.0" : "WSL version: 2.3.26.0");
                return 0;
            }
            if (args.Length == 1 && args[0] == "--status") return 0;
            if (mode == "inventory-timeout" && args.Length == 2 && args[0] == "--list" && args[1] == "--quiet")
            {
                Thread.Sleep(10000);
                return 0;
            }
            // No test should reach install, update, distro launch or UAC.
            return 5;
        }
        if (args.Length == 1 && args[0] == "--version")
        {
            Console.WriteLine("WSL version: 2.6.1.0");
            return 0;
        }
        if (args.Length == 2 && args[1] == "--quiet")
        {
            // Each pipe exceeds its buffer. Sequential draining would deadlock.
            Console.Out.Write(new string('o', 131072));
            Console.Error.Write(new string('e', 131072));
            return 0;
        }
        if (args.Length == 2 && args[1] == "--verbose")
        {
            Thread.Sleep(10000);
            return 0;
        }
        return 5;
    }
}
