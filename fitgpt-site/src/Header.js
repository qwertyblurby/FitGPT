import MountainIcon from './assets/icons/MountainIcon';

function Header() {
	return (
		<header className="flex items-center justify-between px-4 py-6 md:px-6 lg:px-8 bg-gray-950 text-gray-50">
			<div className="flex items-center space-x-4">
				<MountainIcon className="h-8 w-8" />
				<span className="text-xl font-bold">AI Image Classifier</span>
			</div>
			<button className="bg-gray-50 text-gray-950 hover:bg-gray-100 focus-visible:ring-gray-950 dark:bg-gray-800 dark:text-gray-50 dark:hover:bg-gray-700 dark:focus-visible:ring-gray-300">
				Try it Now
			</button>
		</header>
	)
}

export default Header;
