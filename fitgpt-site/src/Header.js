import MountainIcon from './assets/icons/MountainIcon';

function Header() {
	return (
		<header className="flex items-center justify-between px-4 py-6 md:px-6 lg:px-8 bg-gray-950 text-gray-50">
			<div className="flex items-center space-x-4">
				<MountainIcon className="h-8 w-8" />
				<span className="text-xl font-bold">FitGPT</span>
			</div>
			<button className="btn dark:bg-gray-800 dark:text-gray-50 dark:hover:bg-gray-700">
				Try it Now
			</button>
		</header>
	)
}

export default Header;
