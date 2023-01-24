using System.Diagnostics;
using Avalonia.Controls;
using Avalonia.Interactivity;
using AvaloniaMLImageClassifier.ViewModels;

namespace AvaloniaMLImageClassifier.Views
{
    public partial class MainWindow : Window
    {
        private MainWindowViewModel context; 
        public MainWindow()
        {
           
            
            InitializeComponent();
           
            
        }

        private async void Button_OnClick(object? sender, RoutedEventArgs e)
        {
            context = this.DataContext as MainWindowViewModel;
            OpenFileDialog dialog = new OpenFileDialog();
            dialog.Filters.Add(new FileDialogFilter() { Name = "Picture", Extensions =  { "jpg","png" } });
            
            string[] result = await dialog.ShowAsync(this);
            Trace.WriteLine(result[0]);
            context.PathName = result[0];

        }
    }
}