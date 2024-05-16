// import logo from './logo.svg';
import './App.css';
import GoogleFontLoader from 'react-google-font-loader';

function App() {
	return (
		<>
			<GoogleFontLoader
				fonts={[
					{
						font: "Chivo",
						weights: [400, 700],
					},
					{
						font: "Rubik",
						weights: [400, 500, 700],
					}
				]}
			/>
			<div className="App">
				<header className="flex items-center justify-between px-4 py-6 md:px-6 lg:px-8 bg-gray-950 text-gray-50">
					<div className="flex items-center space-x-4">
						<MountainIcon className="h-8 w-8" />
						<span className="text-xl font-bold">AI Image Classifier</span>
					</div>
					<button className="hidden md:inline-flex bg-gray-50 text-gray-950 hover:bg-gray-100 focus-visible:ring-gray-950 dark:bg-gray-800 dark:text-gray-50 dark:hover:bg-gray-700 dark:focus-visible:ring-gray-300">
						Try it Now
					</button>
				</header>
				<main>
					<section className="bg-gray-950 text-gray-50 py-12 md:py-24 lg:py-32">
						<div className="container px-4 md:px-6 lg:px-8">
							<div className="grid items-center gap-6 lg:grid-cols-[1fr_500px] lg:gap-12 xl:grid-cols-[1fr_600px]">
								<div className="space-y-4">
									<h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl">
										Unleash the Power of AI Image Classification
									</h1>
									<p className="max-w-[600px] text-gray-400 md:text-xl">
										Our cutting-edge AI model can accurately classify a wide range of images with lightning-fast speed.
										Experience the future of visual recognition today.
									</p>
									<button className="md:hidden bg-gray-50 text-gray-950 hover:bg-gray-100 focus-visible:ring-gray-950 dark:bg-gray-800 dark:text-gray-50 dark:hover:bg-gray-700 dark:focus-visible:ring-gray-300">
										Try it Now
									</button>
									<div className="flex items-center gap-4">
										<button className="bg-gray-50 text-gray-950 hover:bg-gray-100 focus-visible:ring-gray-950 dark:bg-gray-800 dark:text-gray-50 dark:hover:bg-gray-700 dark:focus-visible:ring-gray-300">
											Upload Image
										</button>
										<input accept="image/*" aria-label="Upload image" className="hidden" type="file" />
									</div>
								</div>
								<img
									alt="Hero"
									className="mx-auto aspect-video overflow-hidden rounded-xl object-cover sm:w-full"
									height="400"
									src={require("./assets/coffindance.png")}
									width="600"
								/>
							</div>
						</div>
					</section>
					
					<section className="py-12 md:py-24 lg:py-32 bg-gray-950 text-gray-50">
						<div className="container px-4 md:px-6 lg:px-8">
							<div className="space-y-6">
								<h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">Capabilities</h2>
								<div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
									<div className="flex flex-col items-center space-y-3">
										<BoltIcon className="h-10 w-10 text-gray-50" />
										<h3 className="text-lg font-semibold">Blazing Fast</h3>
										<p className="text-center text-gray-400">
											Our AI model processes images in milliseconds, delivering lightning-fast results.
										</p>
									</div>
									<div className="flex flex-col items-center space-y-3">
										<MinimizeIcon className="h-10 w-10 text-gray-50" />
										<h3 className="text-lg font-semibold">Efficient Model Architecture</h3>
										<p className="text-center text-gray-400">
											Our AI model is built on a highly optimized and efficient architecture, allowing for fast processing
											and deployment on a wide range of hardware.
										</p>
									</div>
									<div className="flex flex-col items-center space-y-3">
										<ShirtIcon className="h-10 w-10 text-gray-50" />
										<h3 className="text-lg font-semibold">Detailed Predictions</h3>
										<p className="text-center text-gray-400">
											Our AI model provides detailed probability distributions for each article of clothing, giving you a
											comprehensive understanding of the classification results.
										</p>
									</div>
									<div className="flex flex-col items-center space-y-3">
										<PersonStandingIcon className="h-10 w-10 text-gray-50" />
										<h3 className="text-lg font-semibold">Automatic Person Detection</h3>
										<p className="text-center text-gray-400">
											Our advanced computer vision algorithms can automatically detect and segment people within images,
											enabling more accurate classification.
										</p>
									</div>
								</div>
							</div>
						</div>
					</section>
					
					<section className="bg-gray-950 text-gray-50 py-12 md:py-24 lg:py-32">
						<div className="container px-4 md:px-6 lg:px-8">
							<div className="grid items-center gap-6 lg:grid-cols-[1fr_500px] lg:gap-12 xl:grid-cols-[1fr_600px]">
								<img
									alt="Placeholder"
									className="mx-auto aspect-video overflow-hidden rounded-xl object-cover sm:w-full"
									height="400"
									src={require("./assets/coffindance.png")}
									width="600"
								/>
								<div className="space-y-4">
									<h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
										Visualize Classification Results
									</h2>
									<p className="max-w-[600px] text-gray-400 md:text-xl">
										Our AI image classifier provides detailed probability distributions for each class, allowing you to
										understand the model's decision-making process.
									</p>
									<div className="flex flex-col gap-4">
										<div className="flex items-center justify-between">
											<span className="font-medium text-[#FF6B6B]">Dog</span>
											<div className="h-4 w-full rounded-full bg-gray-700">
												<div className="h-4 w-[80%] rounded-full bg-[#FF6B6B]" />
											</div>
											<span className="ml-2 font-medium">80%</span>
										</div>
										<div className="flex items-center justify-between">
											<span className="font-medium text-[#FFC107]">Cat</span>
											<div className="h-4 w-full rounded-full bg-gray-700">
												<div className="h-4 w-[15%] rounded-full bg-[#FFC107]" />
											</div>
											<span className="ml-2 font-medium">15%</span>
										</div>
										<div className="flex items-center justify-between">
											<span className="font-medium text-[#4CAF50]">Bird</span>
											<div className="h-4 w-full rounded-full bg-gray-700">
												<div className="h-4 w-[3%] rounded-full bg-[#4CAF50]" />
											</div>
											<span className="ml-2 font-medium">3%</span>
										</div>
										<div className="flex items-center justify-between">
											<span className="font-medium text-[#673AB7]">Flower</span>
											<div className="h-4 w-full rounded-full bg-gray-700">
												<div className="h-4 w-[2%] rounded-full bg-[#673AB7]" />
											</div>
											<span className="ml-2 font-medium">2%</span>
										</div>
									</div>
								</div>
							</div>
						</div>
					</section>
					
					<section className="bg-gray-950 text-gray-50 py-12 md:py-24 lg:py-32">
						<div className="container px-4 md:px-6 lg:px-8">
							<div className="space-y-6">
								<h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">Technical Details</h2>
								<div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
									<div className="flex flex-col items-center space-y-3">
										<TypeIcon className="h-10 w-10 text-gray-50" />
										<h3 className="text-lg font-semibold">Model Architecture</h3>
										<p className="text-center text-gray-400">
											Our AI model is built using a state-of-the-art deep learning architecture, leveraging the latest
											advancements in computer vision.
										</p>
									</div>
									<div className="flex flex-col items-center space-y-3">
										<TrainTrackIcon className="h-10 w-10 text-gray-50" />
										<h3 className="text-lg font-semibold">Training Dataset</h3>
										<p className="text-center text-gray-400">
											The model has been trained on a diverse and comprehensive dataset, ensuring robust performance
											across a wide range of image categories.
										</p>
									</div>
									<div className="flex flex-col items-center space-y-3">
										<InfoIcon className="h-10 w-10 text-gray-50" />
										<h3 className="text-lg font-semibold">Inference Speed</h3>
										<p className="text-center text-gray-400">
											Our AI model is optimized for lightning-fast inference, processing images in milliseconds and
											delivering real-time results.
										</p>
									</div>
								</div>
							</div>
						</div>
					</section>
					
				</main>
				<footer className="flex flex-col gap-2 sm:flex-row py-6 w-full shrink-0 items-center px-4 md:px-6 border-t bg-gray-950 text-gray-50">
					<p className="text-xs text-gray-400">© 2024 AI Image Classifier. All rights reserved.</p>
					<nav className="sm:ml-auto flex gap-4 sm:gap-6">
						<a href="#" className="text-xs hover:underline underline-offset-4 text-gray-400">Terms of Service</a>
						<a href="#" className="text-xs hover:underline underline-offset-4 text-gray-400">Privacy</a>
					</nav>
				</footer>
				
				
				
				{/* <header className="App-header">
					<img src={logo} className="App-logo" alt="logo" />
					<p>
						Edit <code>src/App.js</code> and save to reload.
					</p>
					<a
						className="App-link"
						href="https://reactjs.org"
						target="_blank"
						rel="noopener noreferrer"
					>
						Learn React
					</a>
				</header> */}
			
			
			</div>
		</>
	);
}

function AccessibilityIcon(props) {
	return (
		<svg
			{...props}
			xmlns="http://www.w3.org/2000/svg"
			width="24"
			height="24"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2"
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<circle cx="16" cy="4" r="1" />
			<path d="m18 19 1-7-6 1" />
			<path d="m5 8 3-3 5.5 3-2.36 3.5" />
			<path d="M4.24 14.5a5 5 0 0 0 6.88 6" />
			<path d="M13.76 17.5a5 5 0 0 0-6.88-6" />
		</svg>
	)
}


function BoltIcon(props) {
	return (
		<svg
			{...props}
			xmlns="http://www.w3.org/2000/svg"
			width="24"
			height="24"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2"
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
			<circle cx="12" cy="12" r="4" />
		</svg>
	)
}


function InfoIcon(props) {
	return (
		<svg
			{...props}
			xmlns="http://www.w3.org/2000/svg"
			width="24"
			height="24"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2"
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<circle cx="12" cy="12" r="10" />
			<path d="M12 16v-4" />
			<path d="M12 8h.01" />
		</svg>
	)
}


function MinimizeIcon(props) {
	return (
		<svg
			{...props}
			xmlns="http://www.w3.org/2000/svg"
			width="24"
			height="24"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2"
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<path d="M8 3v3a2 2 0 0 1-2 2H3" />
			<path d="M21 8h-3a2 2 0 0 1-2-2V3" />
			<path d="M3 16h3a2 2 0 0 1 2 2v3" />
			<path d="M16 21v-3a2 2 0 0 1 2-2h3" />
		</svg>
	)
}


function MountainIcon(props) {
	return (
		<svg
			{...props}
			xmlns="http://www.w3.org/2000/svg"
			width="24"
			height="24"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2"
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<path d="m8 3 4 8 5-5 5 15H2L8 3z" />
		</svg>
	)
}


function PersonStandingIcon(props) {
	return (
		<svg
			{...props}
			xmlns="http://www.w3.org/2000/svg"
			width="24"
			height="24"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2"
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<circle cx="12" cy="5" r="1" />
			<path d="m9 20 3-6 3 6" />
			<path d="m6 8 6 2 6-2" />
			<path d="M12 10v4" />
		</svg>
	)
}


function ShirtIcon(props) {
	return (
		<svg
			{...props}
			xmlns="http://www.w3.org/2000/svg"
			width="24"
			height="24"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2"
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<path d="M20.38 3.46 16 2a4 4 0 0 1-8 0L3.62 3.46a2 2 0 0 0-1.34 2.23l.58 3.47a1 1 0 0 0 .99.84H6v10c0 1.1.9 2 2 2h8a2 2 0 0 0 2-2V10h2.15a1 1 0 0 0 .99-.84l.58-3.47a2 2 0 0 0-1.34-2.23z" />
		</svg>
	)
}


function TrainTrackIcon(props) {
	return (
		<svg
			{...props}
			xmlns="http://www.w3.org/2000/svg"
			width="24"
			height="24"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2"
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<path d="M2 17 17 2" />
			<path d="m2 14 8 8" />
			<path d="m5 11 8 8" />
			<path d="m8 8 8 8" />
			<path d="m11 5 8 8" />
			<path d="m14 2 8 8" />
			<path d="M7 22 22 7" />
		</svg>
	)
}


function TypeIcon(props) {
	return (
		<svg
			{...props}
			xmlns="http://www.w3.org/2000/svg"
			width="24"
			height="24"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2"
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<polyline points="4 7 4 4 20 4 20 7" />
			<line x1="9" x2="15" y1="20" y2="20" />
			<line x1="12" x2="12" y1="4" y2="20" />
		</svg>
	)
}

export default App;
